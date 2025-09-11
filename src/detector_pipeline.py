import os
import shutil
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import io
from PIL import UnidentifiedImageError
import imagehash
import cv2
import pandas as pd
import polars as pl

import stat
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

# Optional: NetworkX (Blossom matching)
try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    _HAS_NX = False

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

    # Alignment(현재 그룹핑엔 미사용, main.py 호환용)
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
    # 우선 OpenCV로 시도
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    # 실패하면 Pillow로 바이트 기반 로드 시도 (경로 인코딩/OneDrive placeholder 문제 완화)
    try:
        with open(path, 'rb') as f:
            data = f.read()
        pil = Image.open(io.BytesIO(data)).convert('L')
        arr = np.array(pil)
        return arr
    except Exception as e:
        raise RuntimeError(f"이미지 로딩 실패: {path} ({e})")


# -------------------------- Prefilters ------------------------------
def phash_of(path: str, roi_ratio: Tuple[float, float, float, float]) -> imagehash.ImageHash:
    try:
        with open(path, 'rb') as f:
            data = f.read()
        img = Image.open(io.BytesIO(data)).convert("L")
    except Exception:
        img = Image.open(path).convert("L")
    img = crop_roi(img, roi_ratio)
    img = img.resize((64, 64))
    return imagehash.phash(img)


def pdq_of(path: str, roi_ratio: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    if not _HAS_PDQ:
        return None
    try:
        with open(path, 'rb') as f:
            data = f.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        img = Image.open(path).convert("RGB")
    img = crop_roi(img, roi_ratio)
    arr = np.array(img)
    hash_vec, _ = pdqhash.compute_pdq_hash(arr)  # 256-d bits (0/1)
    return hash_vec.astype(np.uint8)


def hamming_distance_bits(a_bits: np.ndarray, b_bits: np.ndarray) -> int:
    return int(np.sum(a_bits ^ b_bits))


def ink_density(path: str, roi_ratio: Tuple[float, float, float, float], method: str = "sauvola") -> float:
    gray = read_gray(path)
    h, w = gray.shape[:2]
    l, t, r, b = roi_ratio
    x1, y1, x2, y2 = int(l * w), int(t * h), int(r * w), int(b * h)
    roi = gray[y1:y2, x1:x2]
    if method == "sauvola" and _HAS_SAUVOLA:
        th = threshold_sauvola(roi, window_size=25, k=0.2)
        binary = (roi < th).astype(np.uint8)
    else:
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
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
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
        try:
            with open(p, 'rb') as f:
                data = f.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            img = Image.open(p).convert("RGB")
        img = crop_roi(img, self.roi)
        return self.tf(img), p


def load_model(device: torch.device, backend: str) -> nn.Module:
    if backend == "dinov2" and _HAS_TIMM:
        model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
        model.eval().to(device)
        return model
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
    D_out = (768 if backend == "dinov2" and _HAS_TIMM else 512)
    embs = np.vstack(embs) if len(embs) else np.zeros((0, D_out), dtype=np.float32)
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

    nn = NearestNeighbors(n_neighbors=min(k + 1, N), metric="cosine", algorithm="brute")
    nn.fit(embs)
    dists, idxs = nn.kneighbors(embs, return_distance=True)
    sims = 1.0 - dists
    return idxs, sims, "brute"


# -------------------------- LPIPS / OCR -------------------
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
        # 한국어 손글씨 스캔
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
    return token_set_ratio(a, b) / 100.0


# -------------------------- Utils -------------------
def _handle_remove_readonly(func, path, exc_info):
    # 읽기 전용 파일도 강제 삭제
    os.chmod(path, stat.S_IWRITE)
    func(path)

def _recreate_clean_dir(path: str):
    """폴더를 완전 초기화(삭제 후 재생성, 권한 문제 강제 해제)"""
    if os.path.isdir(path):
        shutil.rmtree(path, onerror=_handle_remove_readonly)
    os.makedirs(path, exist_ok=True)


# -------------------------- Main pipeline ------------------------------
def detect_pipeline(input_dir: str, output_dir: str,
                    config: Optional[DetectorConfig] = None,
                    filter_func: Optional[callable] = None,
                    **_deprecated_kwargs):
    cfg = config or DetectorConfig()

    # ✅ output 폴더 전체를 완전히 삭제 후 재생성 (모든 하위 폴더/파일 초기화)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir, onerror=_handle_remove_readonly)
    os.makedirs(output_dir, exist_ok=True)
    # 하위 폴더도 재생성
    for sub in ["grouped", "ok", "blank_answers", "artifacts"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "artifacts", "thumbnails"), exist_ok=True)

    # 1) Collect image files
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    all_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(exts)]
    if filter_func is not None:
        files = [f for f in all_files if filter_func(f)]
    else:
        files = all_files
    paths = [os.path.join(input_dir, f) for f in files]
    if not files:
        raise FileNotFoundError(f"No images under {input_dir} (필터 적용됨)")

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

    # 5) Pairwise scoring → "확정 유사" 에지 만들기 → (Blossom) 최대가중치매칭으로 2장 그룹화
    print("[4/5] Pair scoring + pairing (max-weight matching) …")

    def prefilter_ok(fi: str, fj: str) -> bool:
        if cfg.prefilter in ("phash", "both"):
            if abs(phashes.get(fi, imagehash.hex_to_hash("0"*16)) - phashes.get(fj, imagehash.hex_to_hash("0"*16))) > cfg.phash_thresh:
                return False
        if cfg.prefilter in ("pdq", "both") and _HAS_PDQ:
            a = pdqs.get(fi, None); b = pdqs.get(fj, None)
            if a is not None and b is not None:
                if hamming_distance_bits(a, b) > cfg.pdq_thresh:
                    return False
        if abs(densities.get(fi, 0.0) - densities.get(fj, 0.0)) > cfg.density_diff_thresh:
            return False
        return True

    all_pair_records: List[Tuple[str, str, float]] = []
    confirmed_edges: List[Tuple[int, int, float]] = []

    # gather all candidate pairs and confirmed edges across all rows
    for i in range(n):
        if idxs.shape[1] == 0:
            continue
        for col in range(1, idxs.shape[1]):  # skip self
            j = int(idxs[i, col])
            if j <= i:
                continue
            fi, fj = name_by_row[i], name_by_row[j]

            # 사용자 요청: 그룹핑은 파일명(확장자 제거) 끝이 '2'인 파일들끼리만 수행
            # 예: 1000652.JPG 와 1000662.JPG 처럼 뒤에 '2'로 끝나는 페어만 그룹화 대상
            if not (os.path.splitext(fi)[0].endswith('2') and os.path.splitext(fj)[0].endswith('2')):
                continue

            # 빈(공백) 이미지는 그룹 대상으로 삼지 않음
            if densities.get(fi, 0.0) <= cfg.blank_density_thresh or densities.get(fj, 0.0) <= cfg.blank_density_thresh:
                continue

            if not prefilter_ok(fi, fj):
                continue

            sim = float(sims[i, col])
            all_pair_records.append((fi, fj, sim))

            confirmed = False
            if sim >= cfg.cnn_thresh:
                confirmed = True
            elif sim >= cfg.suspect_low:
                votes = 0
                if cfg.use_lpips and _HAS_LPIPS:
                    d = lpips_distance(os.path.join(input_dir, fi), os.path.join(input_dir, fj))
                    if d is not None and d <= cfg.lpips_thresh:
                        votes += 1
                if cfg.use_ocr and _HAS_OCR and _HAS_RAPIDFUZZ:
                    ta = texts.get(fi, "") or ocr_text(os.path.join(input_dir, fi))
                    tb = texts.get(fj, "") or ocr_text(os.path.join(input_dir, fj))
                    if text_similarity(ta, tb) >= cfg.text_sim_thresh:
                        votes += 1
                if votes > 0:
                    confirmed = True

            # confirmed 여부가 True이면 edges 목록에 추가
            if confirmed:
                confirmed_edges.append((i, j, sim))

    # --- confirmed_edges로 유사 그래프를 구성하고 연결요소를 그룹으로 추출 ---
    # 모든 confirmed_edges를 수집한 뒤에 한 번만 계산합니다. 여기서는 union-find(Disjoint Set)
    # 을 사용해 더 견고하게 컴포넌트를 추출합니다.
    groups: Dict[str, List[str]] = {}
    gid_counter = 1
    if confirmed_edges:
        # union-find init only for nodes that appear
        parents: Dict[int, int] = {}
        def find(x: int) -> int:
            # path compression
            while parents[x] != x:
                parents[x] = parents[parents[x]]
                x = parents[x]
            return x
        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            parents[rb] = ra

        # initialize parents
        nodes = set()
        for u, v, _w in confirmed_edges:
            nodes.add(u); nodes.add(v)
        for node in nodes:
            parents[node] = node

        # union all edges
        for u, v, _w in confirmed_edges:
            union(u, v)

        # collect groups by root
        comps: Dict[int, List[int]] = {}
        for node in nodes:
            root = find(node)
            comps.setdefault(root, []).append(node)

        for comp_nodes in comps.values():
            if len(comp_nodes) >= 2:
                members = sorted([name_by_row[i] for i in comp_nodes])
                gid = f"group_{gid_counter:03d}"
                groups[gid] = members
                gid_counter += 1

    # 리포트 테이블: 그룹 내 모든 페어를 grouped로 표기
    grouped_pairs_set = set()
    pair_to_gid: Dict[Tuple[str, str], str] = {}
    for gid, members in groups.items():
        # 그룹이 2명 이상일 때 모든 조합을 그룹 페어로 추가
        for a, b in itertools.combinations(members, 2):
            key = tuple(sorted((a, b)))
            grouped_pairs_set.add(key)
            pair_to_gid[key] = gid

    pair_rows: List[List] = []
    for fi, fj, sim in all_pair_records:
        key = tuple(sorted((fi, fj)))
        if key in grouped_pairs_set:
            gid = pair_to_gid.get(key, "-")
            pair_rows.append([fi, fj, round(sim, 4), "중복/그룹", gid])
        else:
            status = "유사 후보" if sim >= cfg.suspect_low else "다름"
            pair_rows.append([fi, fj, round(sim, 4), status, "-"])

    # 6) Save reports / organize outputs
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

    # Copy grouped
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

    # Artifacts (덮어쓰기)
    try:
        np.save(os.path.join(output_dir, "artifacts", "embeddings.npy"), embs)
        with open(os.path.join(output_dir, "artifacts", "ann_backend.txt"), "w", encoding="utf-8") as fw:
            fw.write(backend_used)
    except Exception:
        pass

    return pair_rows, groups


def detect_pipeline_files(file_paths: List[str], output_dir: str,
                          config: Optional[DetectorConfig] = None):
    """
    Similar to detect_pipeline but accepts an explicit list of image file paths.
    file_paths: list of absolute/relative paths to image files.
    """
    cfg = config or DetectorConfig()

    # Normalize and filter existing files
    paths = [os.path.abspath(p) for p in file_paths if os.path.isfile(p)]
    if not paths:
        raise FileNotFoundError("No valid image files provided")

    # Prepare output dirs (same behavior as detect_pipeline)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir, onerror=_handle_remove_readonly)
    os.makedirs(output_dir, exist_ok=True)
    for sub in ["grouped", "ok", "blank_answers", "artifacts"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "artifacts", "thumbnails"), exist_ok=True)

    # files: basenames (used in reports), and a map basename -> full path
    files = [os.path.basename(p) for p in paths]
    path_map = {os.path.basename(p): p for p in paths}

    # The rest of the pipeline expects lists named 'files' and 'paths' where
    # paths are full paths matching files entries. We'll reuse much of the logic
    # from detect_pipeline by reusing variable names.

    # 2) Metadata: prefilters + density + (optional) OCR text
    print(f"[1/5] Metadata (pHash/PDQ + density)")
    phashes: Dict[str, imagehash.ImageHash] = {}
    pdqs: Dict[str, Optional[np.ndarray]] = {}
    densities: Dict[str, float] = {}
    texts: Dict[str, str] = {}

    for f in files:
        p = path_map[f]
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
    embs, ordered_paths = compute_embeddings([path_map[f] for f in files], device, cfg.batch_size, cfg.num_workers, cfg.roi_ratio, cfg.embed_backend)
    n = len(files)
    name_by_row = {i: os.path.basename(ordered_paths[i]) for i in range(n)}

    # 4) ANN candidates
    print("[3/5] Candidate neighbors via ANN …")
    idxs, sims, backend_used = build_candidates(embs, cfg.k, cfg.ann_backend, cfg.hnsw_M, cfg.hnsw_efC, cfg.hnsw_efS)

    # 5) Pairwise scoring → reuse same grouping logic but using path_map when needed
    print("[4/5] Pair scoring + pairing (max-weight matching) …")

    def prefilter_ok(fi: str, fj: str) -> bool:
        if cfg.prefilter in ("phash", "both"):
            if abs(phashes.get(fi, imagehash.hex_to_hash("0"*16)) - phashes.get(fj, imagehash.hex_to_hash("0"*16))) > cfg.phash_thresh:
                return False
        if cfg.prefilter in ("pdq", "both") and _HAS_PDQ:
            a = pdqs.get(fi, None); b = pdqs.get(fj, None)
            if a is not None and b is not None:
                if hamming_distance_bits(a, b) > cfg.pdq_thresh:
                    return False
        if abs(densities.get(fi, 0.0) - densities.get(fj, 0.0)) > cfg.density_diff_thresh:
            return False
        return True

    all_pair_records: List[Tuple[str, str, float]] = []
    confirmed_edges: List[Tuple[int, int, float]] = []

    for i in range(n):
        if idxs.shape[1] == 0:
            continue
        for col in range(1, idxs.shape[1]):
            j = int(idxs[i, col])
            if j <= i:
                continue
            fi, fj = name_by_row[i], name_by_row[j]

            if not (os.path.splitext(fi)[0].endswith('2') and os.path.splitext(fj)[0].endswith('2')):
                continue

            if densities.get(fi, 0.0) <= cfg.blank_density_thresh or densities.get(fj, 0.0) <= cfg.blank_density_thresh:
                continue

            if not prefilter_ok(fi, fj):
                continue

            sim = float(sims[i, col])
            all_pair_records.append((fi, fj, sim))

            confirmed = False
            if sim >= cfg.cnn_thresh:
                confirmed = True
            elif sim >= cfg.suspect_low:
                votes = 0
                if cfg.use_lpips and _HAS_LPIPS:
                    d = lpips_distance(path_map[fi], path_map[fj])
                    if d is not None and d <= cfg.lpips_thresh:
                        votes += 1
                if cfg.use_ocr and _HAS_OCR and _HAS_RAPIDFUZZ:
                    ta = texts.get(fi, "") or ocr_text(path_map[fi])
                    tb = texts.get(fj, "") or ocr_text(path_map[fj])
                    if text_similarity(ta, tb) >= cfg.text_sim_thresh:
                        votes += 1
                if votes > 0:
                    confirmed = True

            if confirmed:
                confirmed_edges.append((i, j, sim))

    # grouping logic (copy from detect_pipeline) — use union-find here as well
    groups: Dict[str, List[str]] = {}
    gid_counter = 1
    if confirmed_edges:
        parents: Dict[int, int] = {}
        def find2(x: int) -> int:
            while parents[x] != x:
                parents[x] = parents[parents[x]]
                x = parents[x]
            return x
        def union2(a: int, b: int):
            ra, rb = find2(a), find2(b)
            if ra == rb:
                return
            parents[rb] = ra

        nodes = set()
        for u, v, _w in confirmed_edges:
            nodes.add(u); nodes.add(v)
        for node in nodes:
            parents[node] = node
        for u, v, _w in confirmed_edges:
            union2(u, v)

        comps: Dict[int, List[int]] = {}
        for node in nodes:
            root = find2(node)
            comps.setdefault(root, []).append(node)
        for comp_nodes in comps.values():
            if len(comp_nodes) >= 2:
                members = sorted([name_by_row[i] for i in comp_nodes])
                gid = f"group_{gid_counter:03d}"
                groups[gid] = members
                gid_counter += 1

    # build pair rows
    grouped_pairs_set = set()
    pair_to_gid: Dict[Tuple[str, str], str] = {}
    for gid, members in groups.items():
        for a, b in itertools.combinations(members, 2):
            key = tuple(sorted((a, b)))
            grouped_pairs_set.add(key)
            pair_to_gid[key] = gid

    pair_rows: List[List] = []
    for fi, fj, sim in all_pair_records:
        key = tuple(sorted((fi, fj)))
        if key in grouped_pairs_set:
            gid = pair_to_gid.get(key, "-")
            pair_rows.append([fi, fj, round(sim, 4), "중복/그룹", gid])
        else:
            status = "유사 후보" if sim >= cfg.suspect_low else "다름"
            pair_rows.append([fi, fj, round(sim, 4), status, "-"])

    # Save reports
    print("[5/5] Save reports / organize outputs …")
    csv_path = os.path.join(output_dir, "report.csv")
    parquet_path = os.path.join(output_dir, "report.parquet")
    df_pairs = pd.DataFrame(pair_rows, columns=["파일1", "파일2", "유사도", "상태", "그룹ID"])
    df_pairs.to_csv(csv_path, index=False, encoding="utf-8-sig")
    pl.from_pandas(df_pairs).write_parquet(parquet_path)

    img_df = pd.DataFrame({
        "파일": files,
        "밀도": [densities.get(f, 0.0) for f in files],
        "빈칸여부": [densities.get(f, 0.0) <= cfg.blank_density_thresh for f in files],
    })
    img_df.to_csv(os.path.join(output_dir, "images_summary.csv"), index=False, encoding="utf-8-sig")

    # Copy grouped / ok / blank
    for gid, members in groups.items():
        gdir = os.path.join(output_dir, "grouped", gid)
        os.makedirs(gdir, exist_ok=True)
        for m in members:
            shutil.copy2(path_map[m], os.path.join(gdir, m))

    okdir = os.path.join(output_dir, "ok")
    bdir = os.path.join(output_dir, "blank_answers")
    os.makedirs(okdir, exist_ok=True)
    os.makedirs(bdir, exist_ok=True)

    grouped_set = set(itertools.chain.from_iterable(groups.values())) if groups else set()
    for f in files:
        src = path_map[f]
        if img_df[img_df["파일"] == f]["빈칸여부"].iloc[0]:
            shutil.copy2(src, os.path.join(bdir, f))
        elif f not in grouped_set:
            shutil.copy2(src, os.path.join(okdir, f))

    try:
        np.save(os.path.join(output_dir, "artifacts", "embeddings.npy"), embs)
        with open(os.path.join(output_dir, "artifacts", "ann_backend.txt"), "w", encoding="utf-8") as fw:
            fw.write(backend_used)
    except Exception:
        pass

    return pair_rows, groups