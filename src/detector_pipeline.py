import os, itertools, shutil
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import imagehash
import cv2

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
    """테두리 제외한 중앙 부분만 추출"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    return img.crop((int(0.15*w), int(0.15*h), int(0.85*w), int(0.85*h)))

def get_embedding(model, path):
    roi = crop_roi(path)
    tensor = transform(roi).unsqueeze(0)
    with torch.no_grad():
        vec = model(tensor).squeeze().numpy()
    return vec

# ---------------- 1차 특징 (pHash + 밀도) ----------------
def get_phash(path):
    img = Image.open(path).convert("L").resize((64, 64))
    return imagehash.phash(img)

def get_density(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, th = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    return np.sum(th > 0) / th.size

# ---------------- 2단계 파이프라인 탐지 ----------------
def detect_pipeline(input_dir, output_dir,
                    phash_thresh=10, density_thresh=0.15,
                    cnn_thresh=0.99, suspect_low=0.95):

    model = load_model()
    files = [f for f in sorted(os.listdir(input_dir))
             if f.lower().endswith((".jpg",".jpeg",".png"))]

    # 1차 특징 계산
    phashes = {f: get_phash(os.path.join(input_dir, f)) for f in files}
    densities = {f: get_density(os.path.join(input_dir, f)) for f in files}

    embeddings = {}
    results, groups = [], {}
    group_id = 1

    for f1, f2 in itertools.combinations(files, 2):
        # --- 1단계: 빠른 필터링 ---
        dist = abs(phashes[f1] - phashes[f2])
        if dist > phash_thresh:
            continue

        dens_diff = abs(densities[f1] - densities[f2])
        if dens_diff > density_thresh:
            continue

        # --- 2단계: 정밀 CNN 비교 ---
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
            if f1 not in groups[g]: groups[g].append(f1)
            if f2 not in groups[g]: groups[g].append(f2)
            results.append([f1, f2, round(sim, 4), "중복/그룹", g])
        elif sim >= suspect_low:
            results.append([f1, f2, round(sim, 4), "유사 후보", "-"])
        else:
            results.append([f1, f2, round(sim, 4), "다름", "-"])

    # 그룹별 폴더 생성 및 파일 복사
    for gid, members in groups.items():
        gdir = os.path.join(output_dir, "grouped", gid)
        os.makedirs(gdir, exist_ok=True)
        for f in members:
            shutil.copy(os.path.join(input_dir, f), os.path.join(gdir, f))

    # 그룹 안 들어간 정상 파일은 ok 폴더로
    okdir = os.path.join(output_dir, "ok")
    os.makedirs(okdir, exist_ok=True)
    for f in files:
        if not any(f in members for members in groups.values()):
            shutil.copy(os.path.join(input_dir, f), os.path.join(okdir, f))

    return results, groups
