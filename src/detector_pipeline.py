import os, itertools, shutil
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import imagehash
import cv2
import pandas as pd
import polars as pl

# ---------------- 모델 초기화 ----------------
def load_model():
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    return model

# ---------------- 전처리 ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def crop_roi(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    return img.crop((int(0.15*w), int(0.15*h), int(0.85*w), int(0.85*h)))

def get_embedding(model, path):
    roi = crop_roi(path)
    tensor = transform(roi).unsqueeze(0)
    with torch.no_grad():
        vec = model(tensor).squeeze().numpy()
    return vec

def get_phash(path):
    img = Image.open(path).convert("L").resize((64, 64))
    return imagehash.phash(img)

def get_density(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, th = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    return np.sum(th > 0) / th.size

# ---------------- 메인 파이프라인 ----------------
def detect_pipeline(input_dir, output_dir,
                    phash_thresh=10, density_thresh=0.15,
                    cnn_thresh=0.99, suspect_low=0.95):

    model = load_model()
    files = [f for f in sorted(os.listdir(input_dir))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    phashes = {f: get_phash(os.path.join(input_dir, f)) for f in files}
    densities = {f: get_density(os.path.join(input_dir, f)) for f in files}
    embeddings = {}
    results, groups = [], {}
    group_id = 1

    for f1, f2 in itertools.combinations(files, 2):
        if abs(phashes[f1] - phashes[f2]) > phash_thresh:
            continue
        if abs(densities[f1] - densities[f2]) > density_thresh:
            continue

        if f1 not in embeddings:
            embeddings[f1] = get_embedding(model, os.path.join(input_dir, f1))
        if f2 not in embeddings:
            embeddings[f2] = get_embedding(model, os.path.join(input_dir, f2))

        sim = cosine_similarity(
            embeddings[f1].reshape(1, -1),
            embeddings[f2].reshape(1, -1)
        )[0][0]

        if sim >= cnn_thresh:
            g = None
            for gid, members in groups.items():
                if f1 in members or f2 in members:
                    g = gid
                    break
            if g is None:
                g = f"group_{group_id:03d}"
                groups[g] = []
                group_id += 1
            for f in (f1, f2):
                if f not in groups[g]:
                    groups[g].append(f)
            results.append([f1, f2, round(sim, 4), "중복/그룹", g])
        elif sim >= suspect_low:
            results.append([f1, f2, round(sim, 4), "유사 후보", "-"])
        else:
            results.append([f1, f2, round(sim, 4), "다름", "-"])

    # 그룹 저장
    for gid, members in groups.items():
        gdir = os.path.join(output_dir, "grouped", gid)
        os.makedirs(gdir, exist_ok=True)
        for f in members:
            shutil.copy(os.path.join(input_dir, f), os.path.join(gdir, f))

    # 정상 파일 저장
    okdir = os.path.join(output_dir, "ok")
    os.makedirs(okdir, exist_ok=True)
    for f in files:
        if not any(f in members for members in groups.values()):
            shutil.copy(os.path.join(input_dir, f), os.path.join(okdir, f))

    # DataFrame 저장 (CSV + Parquet)
    df = pd.DataFrame(results, columns=["파일1","파일2","유사도","상태","그룹ID"])
    csv_path = os.path.join(output_dir, "report.csv")
    parquet_path = os.path.join(output_dir, "report.parquet")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    pl.from_pandas(df).write_parquet(parquet_path)

    return results, groups
