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
import pathlib
import imagehash
import cv2
import pandas as pd
import polars as pl

import stat
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import concurrent.futures
import hashlib

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

# module logger
import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


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
    # 우선 OpenCV로 시도 (빠르고 파일 경로 인코딩 문제에 관대)
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
    except UnidentifiedImageError:
        logger.warning(f"이미지 파싱 실패: {path}")
        return None
    except Exception as e:
        logger.warning(f"이미지 로드 예외: {path} -> {e}")
        return None


# -------------------------- Prefilters ------------------------------
def phash_of(path: str, roi_ratio: Tuple[float, float, float, float]) -> imagehash.ImageHash:
    try:
        img = Image.open(path)
        img = crop_roi(img, roi_ratio)
        return imagehash.phash(img)
    except Exception as e:
        logger.warning(f"phash 계산 실패: {path} -> {e}")
        return None


def pdq_of(path: str, roi_ratio: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    try:
        if not _HAS_PDQ:
            return None
        img = Image.open(path).convert('RGB')
        img = crop_roi(img, roi_ratio)
        arr = np.array(img)
        # pdqhash 라이브러리가 제공하는 API 사용
        if 'pdqhash' in globals():
            # compute_pdq_hash -> (hash_vec, metadata)
            hash_vec, _ = pdqhash.compute_pdq_hash(arr)
            return hash_vec.astype(np.uint8)
        return None
    except Exception as e:
        logger.warning(f"PDQ 계산 실패: {path} -> {e}")
        return None


def hamming_distance_bits(a_bits: np.ndarray, b_bits: np.ndarray) -> int:
    return int(np.sum(a_bits ^ b_bits))


def ink_density(path: str, roi_ratio: Tuple[float, float, float, float], method: str = "sauvola") -> float:
    gray = read_gray(path)
    if gray is None:
        logger.warning(f"ink_density: 이미지 로드 실패로 0 반환: {path}")
        return 0.0
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
    # 안전한 모델 로드: timm 실패 시 ResNet18로 폴백
    try:
        if backend == "dinov2" and _HAS_TIMM:
            model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
            model.eval().to(device)
            return model
    except Exception as e:
        logger.warning(f"timm 모델 로드 실패(backend={backend}): {e}. ResNet18로 폴백합니다.")
    # ResNet 폴백
    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
        model.eval().to(device)
        return model
    except Exception as e:
        logger.error(f"ResNet18 로드 실패: {e}")
        raise


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
            try:
                x = x.to(device)
                out_t = model(x)
                out = out_t.detach().cpu().numpy()
                # flatten spatial dims if present -> (B, D)
                if out.ndim > 2:
                    out = out.reshape(out.shape[0], -1)
                out = out.astype(np.float32)
                embs.append(out)
                ordered_paths.extend(list(pths))
            except Exception as e:
                logger.warning(f"임베딩 배치 처리 실패(일부 배치 건너뜀): {e}")
                continue
    # Stack collected outputs; if none, attempt to infer model output dimensionality
    if len(embs):
        embs = np.vstack(embs)
    else:
        try:
            with torch.no_grad():
                dummy = torch.zeros((1, 3, 224, 224), device=device)
                out = model(dummy).detach().cpu().numpy()
                if out.ndim == 2:
                    D_out = out.shape[1]
                else:
                    D_out = int(np.prod(out.shape[1:]))
        except Exception:
            D_out = (768 if backend == "dinov2" and _HAS_TIMM else 512)
        embs = np.zeros((0, D_out), dtype=np.float32)
    return embs, ordered_paths


# -------------------------- ANN building -----------------------------
def l2_normalize(mat: np.ndarray) -> np.ndarray:
    # 빈 행렬 처리
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # 영 노름 보호: 0으로 나누는 것을 피함
    norms[norms == 0] = 1.0
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
        # 우선순위: FAISS(대규모, 설치됨) -> HNSW -> brute
        if _HAS_FAISS and N >= 2000:
            backend = "faiss"
        elif _HAS_HNSW and N >= 1000:
            backend = "hnsw"
        else:
            backend = "brute"

    # For cosine-based backends normalize once
    use_cosine = backend in ("faiss", "hnsw", "brute")
    mat = embs.astype(np.float32)
    if use_cosine and mat.size:
        mat = l2_normalize(mat)

    if backend == "faiss" and _HAS_FAISS:
        xb = mat
        index = faiss.IndexFlatIP(D)  # inner product == cosine on normalized vectors
        index.add(xb)
        sims, idxs = index.search(xb, min(k + 1, N))
        return idxs, sims, "faiss"

    if backend == "hnsw" and _HAS_HNSW:
        idx = hnswlib.Index(space='cosine', dim=D)
        idx.init_index(max_elements=N, ef_construction=hnsw_efC, M=hnsw_M)
        idx.add_items(mat)
        idx.set_ef(hnsw_efS)
        labels, dists = idx.knn_query(mat, k=min(k + 1, N))
        sims = 1.0 - dists
        return labels, sims, "hnsw"

    nn = NearestNeighbors(n_neighbors=min(k + 1, N), metric="cosine", algorithm="brute")
    nn.fit(mat)
    dists, idxs = nn.kneighbors(mat, return_distance=True)
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
    Araw = cv2.imread(a_path)
    Braw = cv2.imread(b_path)
    if Araw is None or Braw is None:
        logger.warning(f"LPIPS: 이미지 로드 실패 a={a_path} b={b_path}")
        return None
    A = cv2.cvtColor(Araw, cv2.COLOR_BGR2RGB)
    B = cv2.cvtColor(Braw, cv2.COLOR_BGR2RGB)
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


def _copy_to_dir(src: str, dst_dir: str):
    """Copy src file into dst_dir preserving filename; create dst_dir if needed."""
    try:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
        return True
    except Exception as e:
        logger.warning(f"Failed to copy {src} to {dst_dir}: {e}")
        return False


def _safe_recreate_dir(path: str, retries: int = 3, delay: float = 0.5):
    """Try to fully remove and recreate a directory with retries.

    This addresses Windows file-locks or transient permission errors by
    retrying a few times before giving up.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, onerror=_handle_remove_readonly)
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            last_exc = e
            warnings.warn(f"[{attempt}/{retries}] Failed to recreate dir {path}: {e}")
            time.sleep(delay)
    # 마지막 시도 실패
    warnings.warn(f"Could not recreate directory {path} after {retries} attempts: {last_exc}")
    return False


# ---------------------- artifacts / parallel helpers ----------------------
def _max_mtime(paths: List[str]) -> float:
    try:
        return max(os.path.getmtime(p) for p in paths)
    except Exception:
        return 0.0


def _is_fresh(artifact_path: str, paths: List[str]) -> bool:
    if not os.path.exists(artifact_path):
        return False
    try:
        return os.path.getmtime(artifact_path) >= _max_mtime(paths)
    except Exception:
        return False


def _metadata_worker(args):
    # Worker executed in ThreadPoolExecutor for IO-bound metadata tasks
    f, p, cfg = args
    ph = None
    pdq = None
    dens = 0.0
    txt = ""
    try:
        if cfg.prefilter in ("phash", "both"):
            ph = phash_of(p, cfg.roi_ratio)
    except Exception:
        ph = imagehash.hex_to_hash("0" * 16)
    try:
        if cfg.prefilter in ("pdq", "both") and _HAS_PDQ:
            pdq = pdq_of(p, cfg.roi_ratio)
    except Exception:
        pdq = (np.zeros((256,), dtype=np.uint8) if _HAS_PDQ else None)
    try:
        dens = ink_density(p, cfg.roi_ratio, cfg.blank_method)
    except Exception:
        dens = 0.0
    try:
        if cfg.use_ocr and _HAS_OCR and _HAS_RAPIDFUZZ:
            txt = ocr_text(p)
    except Exception:
        txt = ""
    return f, ph, pdq, float(dens), txt


# -------------------------- Main pipeline ------------------------------
def detect_pipeline(input_dir: str, output_dir: str,
                    config: Optional[DetectorConfig] = None,
                    filter_func: Optional[callable] = None,
                    recursive: bool = False,
                    **_deprecated_kwargs):
    cfg = config or DetectorConfig()

    # ✅ output 폴더 전체를 완전히 삭제 후 재생성 (모든 하위 폴더/파일 초기화)
    ok = _safe_recreate_dir(output_dir, retries=5, delay=0.5)
    if not ok:
        # 재시도에도 실패하면 명확한 에러를 던집니다.
        raise RuntimeError(f"Failed to initialize output dir: {output_dir}")
    # 하위 폴더도 재생성
    for sub in ["grouped", "ok", "blank_answers", "artifacts"]:
        subp = os.path.join(output_dir, sub)
        if not _safe_recreate_dir(subp, retries=3, delay=0.2):
            warnings.warn(f"Proceeding despite failing to create subdir: {subp}")
    if not _safe_recreate_dir(os.path.join(output_dir, "artifacts", "thumbnails"), retries=3, delay=0.2):
        warnings.warn("Failed to create thumbnails dir; continuing")

    # 1) Collect image files (optionally recursive)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    all_files = []
    if recursive:
        for root, _, filenames in os.walk(input_dir):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    rel = os.path.relpath(os.path.join(root, fn), input_dir)
                    all_files.append(rel)
        all_files.sort()
    else:
        all_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(exts)]
    if filter_func is not None:
        files = [f for f in all_files if filter_func(f)]
    else:
        files = all_files
    paths = [os.path.join(input_dir, f) for f in files]
    if not files:
        raise FileNotFoundError(f"No images under {input_dir} (필터 적용됨)")

    # 2) Metadata: prefilters + density + (optional) OCR text
    logger.info("[1/5] Metadata (pHash/PDQ + density)")
    phashes: Dict[str, imagehash.ImageHash] = {}
    pdqs: Dict[str, Optional[np.ndarray]] = {}
    densities: Dict[str, float] = {}
    texts: Dict[str, str] = {}

    images_summary_path = os.path.join(output_dir, "images_summary.csv")
    # If existing summary is fresh relative to input files, load densities to skip recompute
    if _is_fresh(images_summary_path, paths):
        try:
            img_df_prev = pd.read_csv(images_summary_path)
            for _, row in img_df_prev.iterrows():
                fname = row.get("파일")
                if fname in files:
                    densities[fname] = float(row.get("밀도", 0.0))
            logger.info("Loaded fresh images_summary.csv -> skipping density recompute for cached entries")
        except Exception:
            pass

    # Prepare worker args and run in ThreadPoolExecutor (IO-bound workloads)
    worker_args = [(f, p, cfg) for f, p in zip(files, paths)]
    # prefer explicit cfg.num_workers when set; otherwise scale reasonably for IO-bound
    if cfg.num_workers and cfg.num_workers > 0:
        max_workers = cfg.num_workers
    else:
        max_workers = min(32, max(4, (os.cpu_count() or 2) * 2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for f, ph, pdqv, dens, txt in ex.map(_metadata_worker, worker_args):
            if ph is not None:
                phashes[f] = ph
            if pdqv is not None:
                pdqs[f] = pdqv
            # only override density if not loaded from fresh summary
            if f not in densities or densities.get(f, 0.0) == 0.0:
                densities[f] = dens
            if txt:
                texts[f] = txt

    # persist images_summary.csv (density + blank flag) for faster subsequent runs
    try:
        img_df = pd.DataFrame({
            "파일": files,
            "밀도": [densities.get(f, 0.0) for f in files],
            "빈칸여부": [densities.get(f, 0.0) <= cfg.blank_density_thresh for f in files],
        })
        img_df.to_csv(images_summary_path, index=False, encoding="utf-8-sig")
    except Exception:
        warnings.warn("Failed to write images_summary.csv")

    # 3) Embeddings
    logger.info("[2/5] CNN/ViT embeddings …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 3) Embeddings: attempt to reuse cached embeddings.npy + ordered_paths.txt
    logger.info("[2/5] CNN/ViT embeddings …")
    emb_art = os.path.join(output_dir, "artifacts", "embeddings.npy")
    opaths_art = os.path.join(output_dir, "artifacts", "ordered_paths.txt")
    embs = None
    ordered_paths = None
    if _is_fresh(emb_art, paths) and os.path.exists(opaths_art):
        try:
            embs = np.load(emb_art)
            with open(opaths_art, "r", encoding="utf-8") as fr:
                ordered_paths = [l.strip() for l in fr.readlines() if l.strip()]
            if len(ordered_paths) != len(files) or embs.shape[0] != len(files):
                logger.warning("Artifact sizes mismatch: forcing re-compute embeddings")
                embs = None
                ordered_paths = None
            else:
                logger.info("Loaded cached embeddings.npy + ordered_paths.txt")
        except Exception as e:
            logger.warning(f"Failed to load embedding artifacts: {e}; will recompute")
            embs = None
            ordered_paths = None

    if embs is None:
        embs, ordered_paths = compute_embeddings(paths, device, cfg.batch_size, cfg.num_workers, cfg.roi_ratio, cfg.embed_backend)
        try:
            np.save(emb_art, embs)
            with open(opaths_art, "w", encoding="utf-8") as fw:
                fw.write("\n".join(ordered_paths))
        except Exception:
            warnings.warn("Failed to save embedding artifacts")

    n = len(files)
    name_by_row = {i: os.path.basename(ordered_paths[i]) for i in range(n)}

    # 4) ANN candidates
    logger.info("[3/5] Candidate neighbors via ANN …")
    idxs, sims, backend_used = build_candidates(embs, cfg.k, cfg.ann_backend, cfg.hnsw_M, cfg.hnsw_efC, cfg.hnsw_efS)

    # 5) Pairwise scoring → "확정 유사" 에지 만들기 → (Blossom) 최대가중치매칭으로 2장 그룹화
    logger.info("[4/5] Pair scoring + pairing (max-weight matching) …")

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
    logger.info("[5/5] Save reports / organize outputs …")
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
        for m in members:
            _copy_to_dir(os.path.join(input_dir, m), gdir)

    okdir = os.path.join(output_dir, "ok")
    bdir = os.path.join(output_dir, "blank_answers")
    os.makedirs(okdir, exist_ok=True)
    os.makedirs(bdir, exist_ok=True)

    grouped_set = set(itertools.chain.from_iterable(groups.values())) if groups else set()
    for f in files:
        src = os.path.join(input_dir, f)
        # 파일명(확장자 제외) 끝 문자에 따라 빈칸 파일의 최종 분류를 조정
        name_wo_ext = os.path.splitext(f)[0]
        if img_df[img_df["파일"] == f]["빈칸여부"].iloc[0]:
            # 기본 동작: blank_answers로 복사
            # 단, 파일명 끝이 '1'이면 blank로 감지되어도 ok로 보관
            if name_wo_ext.endswith('1'):
                dst = os.path.join(okdir, f)
            else:
                dst = os.path.join(bdir, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                _copy_to_dir(src, os.path.dirname(dst))
            except Exception as e:
                logger.warning(f"Failed to copy file {f} (dst={dst}): {e}")
        elif f not in grouped_set:
            dst = os.path.join(okdir, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                _copy_to_dir(src, os.path.dirname(dst))
            except Exception as e:
                logger.warning(f"Failed to copy ok file {f}: {e}")

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
    if not _safe_recreate_dir(output_dir, retries=3, delay=0.2):
        raise RuntimeError(f"Failed to prepare output dir: {output_dir}")
    for sub in ["grouped", "ok", "blank_answers", "artifacts"]:
        if not _safe_recreate_dir(os.path.join(output_dir, sub), retries=2, delay=0.1):
            logger.warning(f"Proceeding despite failing to create subdir: {sub}")
    if not _safe_recreate_dir(os.path.join(output_dir, "artifacts", "thumbnails"), retries=2, delay=0.1):
        logger.warning("Failed to create thumbnails dir; continuing")

    # files: basenames (used in reports), and a map basename -> full path
    files = [os.path.basename(p) for p in paths]
    path_map = {os.path.basename(p): p for p in paths}

    # The rest of the pipeline expects lists named 'files' and 'paths' where
    # paths are full paths matching files entries. We'll reuse much of the logic
    # from detect_pipeline by reusing variable names.

    # 2) Metadata: prefilters + density + (optional) OCR text
    logger.info(f"[1/5] Metadata (pHash/PDQ + density)")
    phashes: Dict[str, imagehash.ImageHash] = {}
    pdqs: Dict[str, Optional[np.ndarray]] = {}
    densities: Dict[str, float] = {}
    texts: Dict[str, str] = {}

    images_summary_path = os.path.join(output_dir, "images_summary.csv")
    if _is_fresh(images_summary_path, paths):
        try:
            img_df_prev = pd.read_csv(images_summary_path)
            for _, row in img_df_prev.iterrows():
                fname = row.get("파일")
                if fname in files:
                    densities[fname] = float(row.get("밀도", 0.0))
            print("Loaded fresh images_summary.csv -> skipping density recompute for cached entries")
        except Exception:
            pass

    worker_args = [(f, path_map[f], cfg) for f in files]
    max_workers = min(32, max(2, (cfg.num_workers or 1) * 4, os.cpu_count() or 2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for f, ph, pdqv, dens, txt in ex.map(_metadata_worker, worker_args):
            if ph is not None:
                phashes[f] = ph
            if pdqv is not None:
                pdqs[f] = pdqv
            if f not in densities or densities.get(f, 0.0) == 0.0:
                densities[f] = dens
            if txt:
                texts[f] = txt

    # 3) Embeddings
    logger.info("[2/5] CNN/ViT embeddings …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_art = os.path.join(output_dir, "artifacts", "embeddings.npy")
    opaths_art = os.path.join(output_dir, "artifacts", "ordered_paths.txt")
    embs = None
    ordered_paths = None
    if _is_fresh(emb_art, paths) and os.path.exists(opaths_art):
        try:
            embs = np.load(emb_art)
            with open(opaths_art, "r", encoding="utf-8") as fr:
                ordered_paths = [l.strip() for l in fr.readlines() if l.strip()]
            if len(ordered_paths) != len(files) or embs.shape[0] != len(files):
                logger.warning("Artifact sizes mismatch: forcing re-compute embeddings")
                embs = None
                ordered_paths = None
            else:
                logger.info("Loaded cached embeddings.npy + ordered_paths.txt")
        except Exception as e:
            logger.warning(f"Failed to load embedding artifacts: {e}; will recompute")
            embs = None
            ordered_paths = None

    if embs is None:
        embs, ordered_paths = compute_embeddings([path_map[f] for f in files], device, cfg.batch_size, cfg.num_workers, cfg.roi_ratio, cfg.embed_backend)
        try:
            np.save(emb_art, embs)
            with open(opaths_art, "w", encoding="utf-8") as fw:
                fw.write("\n".join(ordered_paths))
        except Exception:
            warnings.warn("Failed to save embedding artifacts")

    n = len(files)
    name_by_row = {i: os.path.basename(ordered_paths[i]) for i in range(n)}

    # 4) ANN candidates
    logger.info("[3/5] Candidate neighbors via ANN …")
    idxs, sims, backend_used = build_candidates(embs, cfg.k, cfg.ann_backend, cfg.hnsw_M, cfg.hnsw_efC, cfg.hnsw_efS)

    # 5) Pairwise scoring → reuse same grouping logic but using path_map when needed
    logger.info("[4/5] Pair scoring + pairing (max-weight matching) …")

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
    logger.info("[5/5] Save reports / organize outputs …")
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
        for m in members:
            _copy_to_dir(path_map[m], gdir)

    okdir = os.path.join(output_dir, "ok")
    bdir = os.path.join(output_dir, "blank_answers")
    os.makedirs(okdir, exist_ok=True)
    os.makedirs(bdir, exist_ok=True)

    grouped_set = set(itertools.chain.from_iterable(groups.values())) if groups else set()
    for f in files:
        src = path_map[f]
        name_wo_ext = os.path.splitext(f)[0]
        if img_df[img_df["파일"] == f]["빈칸여부"].iloc[0]:
            # 기본적으로 blank_answers에는 '*2'로 끝나는 파일만 넣도록 하되,
            # 파일명 끝이 '1'이면 blank로 감지되어도 ok로 분류합니다.
            if name_wo_ext.endswith('2'):
                try:
                    _copy_to_dir(src, os.path.join(bdir))
                except Exception as e:
                    logger.warning(f"Failed to copy blank answer {f}: {e}")
            elif name_wo_ext.endswith('1'):
                try:
                    _copy_to_dir(src, os.path.join(okdir))
                except Exception as e:
                    logger.warning(f"Failed to copy reclassified ok file {f}: {e}")
            else:
                # 그 외의 빈칸 감지 파일은 원래대로 복사하지 않음
                pass
        elif f not in grouped_set:
            try:
                _copy_to_dir(src, os.path.join(okdir))
            except Exception as e:
                logger.warning(f"Failed to copy ok file {f}: {e}")

    try:
        np.save(os.path.join(output_dir, "artifacts", "embeddings.npy"), embs)
        with open(os.path.join(output_dir, "artifacts", "ann_backend.txt"), "w", encoding="utf-8") as fw:
            fw.write(backend_used)
    except Exception:
        pass

    return pair_rows, groups