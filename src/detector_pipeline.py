import os
import math
import shutil
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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

# Optional FAISS import (fallback to sklearn if not available)
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from sklearn.neighbors import NearestNeighbors


@dataclass
class DetectorConfig:
    use_faiss: bool = False
    k: int = 20
    phash_thresh: int = 10
    density_diff_thresh: float = 0.15
    cnn_thresh: float = 0.99
    suspect_low: float = 0.95
    blank_density_thresh: float = 0.02
    batch_size: int = 64
    num_workers: int = 0
    roi_ratio: Tuple[float, float, float, float] = (0.15, 0.15, 0.85, 0.85)  # l,t,r,b


# -------------------------- Utility: ROI crop --------------------------
def crop_roi(img: Image.Image, roi_ratio: Tuple[float, float, float, float]):
    w, h = img.size
    l, t, r, b = roi_ratio
    return img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


# -------------------------- Metadata (pHash, density) ------------------
def phash_of(path: str, roi_ratio: Tuple[float, float, float, float]) -> imagehash.ImageHash:
    img = Image.open(path).convert("L")
    img = crop_roi(img, roi_ratio)
    img = img.resize((64, 64))
    return imagehash.phash(img)


def ink_density_of(path: str, roi_ratio: Tuple[float, float, float, float]) -> float:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    l, t, r, b = roi_ratio
    x1, y1, x2, y2 = int(l * w), int(t * h), int(r * w), int(b * h)
    roi = img[y1:y2, x1:x2]
    # Otsu binary for robustness (space/연필흔적도 안정적으로 반응)
    _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    density = float(np.count_nonzero(th)) / th.size
    return density


# -------------------------- Torch Dataset -----------------------------
class ImgDataset(Dataset):
    def __init__(self, paths: List[str], roi_ratio: Tuple[float, float, float, float]):
        self.paths = paths
        self.roi = roi_ratio
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = crop_roi(img, self.roi)
        return self.tf(img), p


# -------------------------- Model / Embeddings -------------------------
def load_model(device: torch.device) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()  # 512-d embedding
    model.eval()
    model.to(device)
    return model


def compute_embeddings(paths: List[str], device: torch.device, batch_size: int, num_workers: int,
                        roi_ratio: Tuple[float, float, float, float]):
    ds = ImgDataset(paths, roi_ratio)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    model = load_model(device)

    embs = []
    ordered_paths = []
    with torch.no_grad():
        for x, pths in dl:
            x = x.to(device)
            out = model(x).detach().cpu().numpy().astype(np.float32)
            embs.append(out)
            ordered_paths.extend(list(pths))
    embs = np.vstack(embs) if len(embs) else np.zeros((0, 512), dtype=np.float32)
    return embs, ordered_paths


# -------------------------- Index & Candidate graph --------------------
def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def build_candidates(embs: np.ndarray, k: int, force_faiss: bool):
    """
    Return (indices, sims, used_faiss).
    Indices shape (N, k+1), sims shape (N, k+1) including self at column 0.
    """
    N, D = embs.shape
    if N == 0:
        return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=np.float32), False

    if _HAS_FAISS and (force_faiss or N >= 2000):
        xb = l2_normalize(embs.astype(np.float32))
        index = faiss.IndexFlatIP(D)  # inner product == cosine on normalized vectors
        index.add(xb)
        sims, idxs = index.search(xb, min(k + 1, N))  # include self at col 0
        return idxs, sims, True
    else:
        nn = NearestNeighbors(n_neighbors=min(k + 1, N), metric="cosine", algorithm="brute")
        nn.fit(embs)
        dists, idxs = nn.kneighbors(embs, return_distance=True)
        sims = 1.0 - dists  # cosine similarity
        return idxs, sims, False


# -------------------------- DSU (Union-Find) ---------------------------
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


# -------------------------- Main pipeline ------------------------------
def detect_pipeline(input_dir: str, output_dir: str,
                    config: Optional[DetectorConfig] = None,
                    phash_thresh: Optional[int] = None,
                    density_thresh: Optional[float] = None,
                    cnn_thresh: Optional[float] = None,
                    suspect_low: Optional[float] = None):
    """
    Backward-compatible signature with keyword overrides.
    Returns: (pair_results_list, groups_dict)
      - pair_results_list: List[[파일1, 파일2, 유사도, 상태, 그룹ID]]
      - groups_dict: {group_id: [filenames...]}
    """
    cfg = config or DetectorConfig()
    # Back-compat overrides
    if phash_thresh is not None:
        cfg.phash_thresh = phash_thresh
    if density_thresh is not None:
        cfg.density_diff_thresh = density_thresh
    if cnn_thresh is not None:
        cfg.cnn_thresh = cnn_thresh
    if suspect_low is not None:
        cfg.suspect_low = suspect_low

    os.makedirs(output_dir, exist_ok=True)
    for sub in ["grouped", "ok", "blank_answers", "artifacts"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    # 1) Collect image files
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(exts)]
    paths = [os.path.join(input_dir, f) for f in files]
    if not files:
        raise FileNotFoundError(f"No images under {input_dir}")

    # 2) Compute metadata: pHash & density
    print(f"[1/4] Metadata (pHash, density) for {len(files)} images…")
    phashes: Dict[str, imagehash.ImageHash] = {}
    densities: Dict[str, float] = {}
    for f, p in zip(files, paths):
        try:
            phashes[f] = phash_of(p, cfg.roi_ratio)
            densities[f] = ink_density_of(p, cfg.roi_ratio)
        except Exception as e:
            warnings.warn(f"Metadata failed for {f}: {e}")
            phashes[f] = imagehash.hex_to_hash("0" * 16)
            densities[f] = 0.0

    # 3) Embeddings
    print("[2/4] CNN embeddings (ResNet18) …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embs, ordered_paths = compute_embeddings(paths, device, cfg.batch_size, cfg.num_workers, cfg.roi_ratio)
    n = len(files)
    name_by_row = {i: os.path.basename(ordered_paths[i]) for i in range(n)}

    # 4) Candidate neighbors via FAISS/KNN
    print("[3/4] Candidate neighbors via FAISS/KNN …")
    idxs, sims, used_faiss = build_candidates(embs, cfg.k, cfg.use_faiss)

    # 5) Pairwise scoring + grouping via DSU
    print("[4/4] Pair scoring + grouping …")
    dsu = DSU(n)
    pair_rows: List[List] = []

    for i in range(n):
        if idxs.shape[1] == 0:
            continue
        for col in range(1, idxs.shape[1]):  # skip self at col 0
            j = int(idxs[i, col])
            if j <= i:
                continue  # avoid duplicates
            fi, fj = name_by_row[i], name_by_row[j]

            # Prefilter with pHash and density difference
            if abs(phashes[fi] - phashes[fj]) > cfg.phash_thresh:
                continue
            if abs(densities[fi] - densities[fj]) > cfg.density_diff_thresh:
                continue

            sim = float(sims[i, col])
            status = "다름"
            gid = "-"

            if sim >= cfg.cnn_thresh:
                dsu.union(i, j)
                status = "중복/그룹"
            elif sim >= cfg.suspect_low:
                status = "유사 후보"

            pair_rows.append([fi, fj, round(sim, 4), status, gid])

    # Finalize groups from DSU
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

    # Update group ids in pair_rows for grouped pairs
    group_of = {}
    for gid, members in groups.items():
        for m in members:
            group_of[m] = gid
    for row in pair_rows:
        if row[3] == "중복/그룹":
            fi, fj = row[0], row[1]
            gid = group_of.get(fi) or group_of.get(fj) or "-"
            row[4] = gid

    # 6) Write outputs
    csv_path = os.path.join(output_dir, "report.csv")
    parquet_path = os.path.join(output_dir, "report.parquet")
    df_pairs = pd.DataFrame(pair_rows, columns=["파일1", "파일2", "유사도", "상태", "그룹ID"])
    df_pairs.to_csv(csv_path, index=False, encoding="utf-8-sig")
    pl.from_pandas(df_pairs).write_parquet(parquet_path)

    # Image summary (extra)
    img_df = pd.DataFrame({
        "파일": files,
        "밀도": [densities[f] for f in files],
        "빈칸여부": [densities[f] <= cfg.blank_density_thresh for f in files],
    })
    img_df.to_csv(os.path.join(output_dir, "images_summary.csv"), index=False, encoding="utf-8-sig")

    # 7) Copy images to grouped / ok / blank_answers
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

    # Artifact: save normalized embeddings for debug/analysis
    try:
        np.save(os.path.join(output_dir, "artifacts", "embeddings.npy"), embs)
    except Exception:
        pass

    return pair_rows, groups
