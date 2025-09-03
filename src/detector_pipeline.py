import os
import shutil
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from components.magnifier import magnifier

import numpy as np
from PIL import Image
import imagehash
import cv2
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# Optional: timm (DINOv2)
try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

# Optional: FAISS
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# Optional: HNSW
try:
    import hnswlib
    _HAS_HNSW = True
except Exception:
    _HAS_HNSW = False

# Optional: LPIPS
try:
    import lpips
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False

# Optional: PDQ hash
try:
    import pdqhash  # may be missing on some platforms
    _HAS_PDQ = True
except Exception:
    _HAS_PDQ = False

# Optional: OCR + RapidFuzz
try:
    from paddleocr import PaddleOCR
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

try:
    from rapidfuzz.fuzz import token_set_ratio
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# Optional: Sauvola
try:
    from skimage.filters import threshold_sauvola
    _HAS_SAUVOLA = True
except Exception:
    _HAS_SAUVOLA = False

from sklearn.neighbors import NearestNeighbors


@dataclass
class DetectorConfig:
    # Backends
    embed_backend: str = "dinov2"  # or resnet18
    ann_backend: str = "auto"       # auto/brute/faiss/hnsw

    # ANN params
    k: int = 20
    hnsw_M: int = 32
    hnsw_efC: int = 200
    hnsw_efS: int = 64

    # Prefilters
    prefilter: str = "phash"        # phash/pdq/both
    phash_thresh: int = 10
    pdq_thresh: int = 80
    density_diff_thresh: float = 0.15

    # Similarity thresholds
    cnn_thresh: float = 0.99
    suspect_low: float = 0.95

    # Blank detection
    blank_method: str = "sauvola"   # otsu/sauvola
    blank_density_thresh: float = 0.02

    # Re-ranking / OCR
    use_lpips: bool = False
    lpips_thresh: float = 0.2
    use_ocr: bool = False
    text_sim_thresh: float = 0.85

    # Alignment
    use_alignment: bool = False

    # Embedding
    batch_size: int = 64
    num_workers: int = 0
    roi_ratio: Tuple[float, float, float, float] = (0.15, 0.15, 0.85, 0.85)


# -------------------------- ROI / Helpers --------------------------
def crop_roi(img: Image.Image, roi_ratio: Tuple[float, float, float, float]):
    w, h = img.size
    l, t, r, b = roi_ratio
    return img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


def read_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"이미지 로딩 실패: {path}")
    return img


# -------------------------- Prefilters ------------------------------
def phash_of(path: str, roi_ratio: Tuple[float, float, float, float]) -> imagehash.ImageHash:
    img = Image.open(path).convert("L")
    img = crop_roi(img, roi_ratio)
    img = img.resize((64, 64))
    return imagehash.phash(img)


def pdq_of(path: str, roi_ratio: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    if not _HAS_PDQ:
        return None
    img = Image.open(path).convert("RGB")
    img = crop_roi(img, roi_ratio)
    arr = np.array(img)
    hash_vec, _ = pdqhash.compute_pdq_hash(arr)  # returns 256-d bits (np.array of 0/1)
    return hash_vec.astype(np.uint8)


def hamming_distance_bits(a_bits: np.ndarray, b_bits: np.ndarray) -> int:
    # expects uint8 {0,1}
    return int(np.sum(a_bits ^ b_bits))


def ink_density(path: str, roi_ratio: Tuple[float, float, float, float], method: str = "sauvola") -> float:
    """Return foreground ratio in ROI using binarization."""
    gray = read_gray(path)
    h, w = gray.shape[:2]
    l, t, r, b = roi_ratio
    x1, y1, x2, y2 = int(l * w), int(t * h), int(r * w), int(b * h)
    roi = gray[y1:y2, x1:x2]
    if method == "sauvola" and _HAS_SAUVOLA:
        th = threshold_sauvola(roi, window_size=25, k=0.2)
        binary = (roi < th).astype(np.uint8)
    else:
        # Otsu fallback
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = (binary > 0).astype(np.uint8)
    return float(np.count_nonzero(binary)) / binary.size


# -------------------------- Dataset / Embedding ----------------------
class ImgDataset(Dataset):
    def __init__(self, paths: List[str], roi_ratio: Tuple[float, float, float, float], backend: str):
        self.paths = paths
        self.roi = roi_ratio
        self.backend = backend
        if backend == "dinov2":
            # DINOv2 추천 전처리
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            # ResNet18
            self.tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = crop_roi(img, self.roi)
        return self.tf(img), p


def load_model(device: torch.device, backend: str) -> nn.Module:
    if backend == "dinov2" and _HAS_TIMM:
        model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
        model.eval().to(device)
        return model
    # fallback: resnet18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)
    return model


def compute_embeddings(paths: List[str], device: torch.device, batch_size: int, num_workers: int,
                       roi_ratio: Tuple[float, float, float, float], backend: str):
    ds = ImgDataset(paths, roi_ratio, backend)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    model = load_model(device, backend)

    embs = []
    ordered_paths = []
    with torch.no_grad():
        for x, pths in dl:
            x = x.to(device)
            out = model(x).detach().cpu().numpy().astype(np.float32)
            embs.append(out)
            ordered_paths.extend(list(pths))
    embs = np.vstack(embs) if len(embs) else np.zeros((0, 768 if backend == "dinov2" and _HAS_TIMM else 512), dtype=np.float32)
    return embs, ordered_paths


# -------------------------- ANN building -----------------------------
def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def build_candidates(embs: np.ndarray, k: int, ann_backend: str,
                     hnsw_M: int, hnsw_efC: int, hnsw_efS: int):
    """
    Return (indices, sims, backend_used).
    Indices shape (N, k+1), sims shape (N, k+1) including self at column 0.
    """
    N, D = embs.shape
    if N == 0:
        return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=np.float32), "none"

    # auto choose
    backend = ann_backend
    if ann_backend == "auto":
        if _HAS_HNSW and N >= 1000:
            backend = "hnsw"
        elif _HAS_FAISS and N >= 2000:
            backend = "faiss"
        else:
            backend = "brute"

    if backend == "faiss" and _HAS_FAISS:
        xb = l2_normalize(embs.astype(np.float32))
        index = faiss.IndexFlatIP(D)  # inner product == cosine on normalized vectors
        index.add(xb)
        sims, idxs = index.search(xb, min(k + 1, N))
        return idxs, sims, "faiss"

    if backend == "hnsw" and _HAS_HNSW:
        idx = hnswlib.Index(space='cosine', dim=D)
        idx.init_index(max_elements=N, ef_construction=hnsw_efC, M=hnsw_M)
        idx.add_items(embs)
        idx.set_ef(hnsw_efS)
        labels, dists = idx.knn_query(embs, k=min(k + 1, N))
        sims = 1.0 - dists
        return labels, sims, "hnsw"

    # brute (sklearn)
    nn = NearestNeighbors(n_neighbors=min(k + 1, N), metric="cosine", algorithm="brute")
    nn.fit(embs)
    dists, idxs = nn.kneighbors(embs, return_distance=True)
    sims = 1.0 - dists
    return idxs, sims, "brute"


# -------------------------- LPIPS / OCR / Alignment -------------------
_lpips_model = None
def lpips_distance(a_path: str, b_path: str) -> Optional[float]:
    global _lpips_model
    if not _HAS_LPIPS:
        return None
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='vgg').eval()
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor()])
    A = cv2.cvtColor(cv2.imread(a_path), cv2.COLOR_BGR2RGB)
    B = cv2.cvtColor(cv2.imread(b_path), cv2.COLOR_BGR2RGB)
    h = min(A.shape[0], B.shape[0]); w = min(A.shape[1], B.shape[1])
    A = cv2.resize(A, (w, h)); B = cv2.resize(B, (w, h))
    a = tf(Image.fromarray(A)).unsqueeze(0)
    b = tf(Image.fromarray(B)).unsqueeze(0)
    with torch.no_grad():
        d = _lpips_model(a, b).item()
    return float(d)


_ocr = None
def ocr_text(path: str) -> str:
    global _ocr
    if not (_HAS_OCR and _HAS_RAPIDFUZZ):
        return ""
    if _ocr is None:
        # 한국어 손글씨 스캔: 'korean' 또는 'korean+english' 선택
        _ocr = PaddleOCR(lang='korean', use_angle_cls=True, show_log=False)
    res = _ocr.ocr(path, cls=True)
    texts = []
    try:
        for line in res[0]:
            texts.append(line[1][0])
    except Exception:
        pass
    return " ".join(texts)


def text_similarity(a: str, b: str) -> float:
    if not _HAS_RAPIDFUZZ:
        return 0.0
    # token_set_ratio → 0..100
    return token_set_ratio(a, b) / 100.0


def align_pair(a_path: str, b_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """ECC 정렬(affine). 실패 시 동일 크기로 리사이즈만."""
    A = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
    B = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
    if A is None or B is None:
        raise RuntimeError("이미지 로딩 실패")
    h = min(A.shape[0], B.shape[0]); w = min(A.shape[1], B.shape[1])
    A = cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA)
    B = cv2.resize(B, (w, h), interpolation=cv2.INTER_AREA)

    # ECC requires float32, normalized
    A_f = A.astype(np.float32) / 255.0
    B_f = B.astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
        cc, warp = cv2.findTransformECC(A_f, B_f, warp, cv2.MOTION_AFFINE, criteria, None, 5)
        B_aligned = cv2.warpAffine(B, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return A, B_aligned
    except Exception:
        # fallback: no alignment, just resized
        return A, B


# -------------------------- Main pipeline ------------------------------
def detect_pipeline(input_dir: str, output_dir: str,
                    config: Optional[DetectorConfig] = None,
                    **_deprecated_kwargs):
    cfg = config or DetectorConfig()

    os.makedirs(output_dir, exist_ok=True)
    for sub in ["grouped", "ok", "blank_answers", "artifacts"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    # 1) Collect image files
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(exts)]
    paths = [os.path.join(input_dir, f) for f in files]
    if not files:
        raise FileNotFoundError(f"No images under {input_dir}")

    # 2) Metadata: prefilters + density + (optional) OCR text
    print(f"[1/5] Metadata (pHash/PDQ + density)")
    phashes: Dict[str, imagehash.ImageHash] = {}
    pdqs: Dict[str, Optional[np.ndarray]] = {}
    densities: Dict[str, float] = {}
    texts: Dict[str, str] = {}

    for f, p in zip(files, paths):
        try:
            if cfg.prefilter in ("phash", "both"):
                phashes[f] = phash_of(p, cfg.roi_ratio)
            if cfg.prefilter in ("pdq", "both") and _HAS_PDQ:
                pdqs[f] = pdq_of(p, cfg.roi_ratio)
            densities[f] = ink_density(p, cfg.roi_ratio, cfg.blank_method)
            if cfg.use_ocr and _HAS_OCR and _HAS_RAPIDFUZZ:
                texts[f] = ocr_text(p)
        except Exception as e:
            warnings.warn(f"Metadata failed for {f}: {e}")
            if cfg.prefilter in ("phash", "both"):
                phashes[f] = imagehash.hex_to_hash("0" * 16)
            if cfg.prefilter in ("pdq", "both") and _HAS_PDQ:
                pdqs[f] = np.zeros((256,), dtype=np.uint8)
            densities[f] = 0.0
            if cfg.use_ocr:
                texts[f] = ""

    # 3) Embeddings
    print("[2/5] CNN/ViT embeddings …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embs, ordered_paths = compute_embeddings(paths, device, cfg.batch_size, cfg.num_workers, cfg.roi_ratio, cfg.embed_backend)
    n = len(files)
    name_by_row = {i: os.path.basename(ordered_paths[i]) for i in range(n)}

    # 4) ANN candidates
    print("[3/5] Candidate neighbors via ANN …")
    idxs, sims, backend_used = build_candidates(embs, cfg.k, cfg.ann_backend, cfg.hnsw_M, cfg.hnsw_efC, cfg.hnsw_efS)
    # 5) Pairwise scoring + optional re-ranking + grouping
    print("[4/5] Pair scoring + (optional) re-ranking + grouping …")
    # Union-Find
    class DSU:
        def __init__(self, n: int):
            self.p = list(range(n))
            self.sz = [1] * n
        def find(self, x: int) -> int:
            while self.p[x] != x:
                self.p[x] = self.p[self.p[x]]
                x = self.p[x]
            return x
        def union(self, a: int, b: int):
            ra, rb = self.find(a), self.find(b)
            if ra == rb:
                return
            if self.sz[ra] < self.sz[rb]:
                ra, rb = rb, ra
            self.p[rb] = ra
            self.sz[ra] += self.sz[rb]

    dsu = DSU(n)
    pair_rows: List[List] = []

    def prefilter_ok(fi: str, fj: str) -> bool:
        ok = True
        if cfg.prefilter in ("phash", "both"):
            if abs(phashes.get(fi, imagehash.hex_to_hash("0"*16)) - phashes.get(fj, imagehash.hex_to_hash("0"*16))) > cfg.phash_thresh:
                return False
        if cfg.prefilter in ("pdq", "both") and _HAS_PDQ:
            a = pdqs.get(fi, None); b = pdqs.get(fj, None)
            if a is None or b is None:
                pass
            else:
                if hamming_distance_bits(a, b) > cfg.pdq_thresh:
                    return False
        if abs(densities.get(fi, 0.0) - densities.get(fj, 0.0)) > cfg.density_diff_thresh:
            return False
        return ok

    for i in range(n):
        if idxs.shape[1] == 0:
            continue
        for col in range(1, idxs.shape[1]):  # skip self
            j = int(idxs[i, col])
            if j <= i:
                continue
            fi, fj = name_by_row[i], name_by_row[j]

            # Prefilter
            if not prefilter_ok(fi, fj):
                continue

            sim = float(sims[i, col])
            status = "다름"
            gid = "-"

            confirmed = False
            if sim >= cfg.cnn_thresh:
                confirmed = True
            elif sim >= cfg.suspect_low:
                # Re-ranking (optional)
                rank_votes = 0
                votes_need = 1  # 단일 기준으로도 업그레이드 가능
                if cfg.use_lpips and _HAS_LPIPS:
                    d = lpips_distance(os.path.join(input_dir, fi), os.path.join(input_dir, fj))
                    if d is not None and d <= cfg.lpips_thresh:
                        rank_votes += 1
                if cfg.use_ocr and _HAS_OCR and _HAS_RAPIDFUZZ:
                    ta = texts.get(fi, "") or ocr_text(os.path.join(input_dir, fi))
                    tb = texts.get(fj, "") or ocr_text(os.path.join(input_dir, fj))
                    ts = text_similarity(ta, tb)
                    if ts >= cfg.text_sim_thresh:
                        rank_votes += 1
                if rank_votes >= votes_need:
                    confirmed = True
                    status = "중복/그룹"
                else:
                    status = "유사 후보"

            if confirmed:
                dsu.union(i, j)
                status = "중복/그룹" if status != "유사 후보" else "중복/그룹"

            pair_rows.append([fi, fj, round(sim, 4), status, gid])

    # Finalize groups
    groups: Dict[str, List[str]] = {}
    root_to_members: Dict[int, List[str]] = {}
    for i in range(n):
        r = dsu.find(i)
        root_to_members.setdefault(r, []).append(name_by_row[i])

    gid_counter = 1
    for r, members in root_to_members.items():
        if len(members) >= 2:
            gid = f"group_{gid_counter:03d}"
            groups[gid] = sorted(members)
            gid_counter += 1

    # Update group ids in pair_rows
    group_of = {}
    for gid, members in groups.items():
        for m in members:
            group_of[m] = gid
    for row in pair_rows:
        if row[3] == "중복/그룹":
            fi, fj = row[0], row[1]
            gid = group_of.get(fi) or group_of.get(fj) or "-"
            row[4] = gid

    # 6) Write outputs + copy images by class
    print("[5/5] Save reports / organize outputs …")
    csv_path = os.path.join(output_dir, "report.csv")
    parquet_path = os.path.join(output_dir, "report.parquet")
    df_pairs = pd.DataFrame(pair_rows, columns=["파일1", "파일2", "유사도", "상태", "그룹ID"])
    df_pairs.to_csv(csv_path, index=False, encoding="utf-8-sig")
    pl.from_pandas(df_pairs).write_parquet(parquet_path)

    # Image summary (density/blank)
    img_df = pd.DataFrame({
        "파일": files,
        "밀도": [densities.get(f, 0.0) for f in files],
        "빈칸여부": [densities.get(f, 0.0) <= cfg.blank_density_thresh for f in files],
    })
    img_df.to_csv(os.path.join(output_dir, "images_summary.csv"), index=False, encoding="utf-8-sig")

    # Copy
    for gid, members in groups.items():
        gdir = os.path.join(output_dir, "grouped", gid)
        os.makedirs(gdir, exist_ok=True)
        for m in members:
            shutil.copy2(os.path.join(input_dir, m), os.path.join(gdir, m))

    okdir = os.path.join(output_dir, "ok")
    bdir = os.path.join(output_dir, "blank_answers")
    os.makedirs(okdir, exist_ok=True)
    os.makedirs(bdir, exist_ok=True)

    grouped_set = set(itertools.chain.from_iterable(groups.values())) if groups else set()
    for f in files:
        src = os.path.join(input_dir, f)
        if img_df[img_df["파일"] == f]["빈칸여부"].iloc[0]:
            shutil.copy2(src, os.path.join(bdir, f))
        elif f not in grouped_set:
            shutil.copy2(src, os.path.join(okdir, f))

    # Artifacts
    try:
        np.save(os.path.join(output_dir, "artifacts", "embeddings.npy"), embs)
        with open(os.path.join(output_dir, "artifacts", "ann_backend.txt"), "w", encoding="utf-8") as fw:
            fw.write(backend_used)
    except Exception:
        pass

    return pair_rows, groups

    