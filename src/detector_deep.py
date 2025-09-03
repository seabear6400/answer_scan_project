import os
import itertools
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- 모델 초기화 ----------------
def load_model():
    """ResNet18을 feature extractor로 불러오기"""
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()  # 마지막 분류층 제거 -> feature vector 추출
    model.eval()
    return model

# ---------------- 이미지 전처리 ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def crop_roi(path):
    """
    답안지 전체가 아니라 글씨가 있을 가능성이 높은 '중앙 부분'만 자르기.
    테두리, 여백, 공통 레이아웃 영향 줄이고 손글씨 반영 ↑
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # 중앙 80% 영역만 사용 (상단/하단/좌우 10% 잘라냄)
    roi = img.crop((int(0.1 * w), int(0.1 * h), int(0.9 * w), int(0.9 * h)))
    return roi

def get_embedding(model, path):
    """이미지에서 임베딩 벡터 추출"""
    roi = crop_roi(path)
    tensor = transform(roi).unsqueeze(0)
    with torch.no_grad():
        vec = model(tensor).squeeze().numpy()
    return vec

# ---------------- 중복 탐지 ----------------
def detect_duplicates_deep(input_dir, threshold=0.99, suspect_low=0.95):
    """
    딥러닝 임베딩 기반 중복 탐지
    - threshold 이상: 확실한 중복 (재스캔 필요)
    - suspect_low 이상: 의심 후보 (검수 필요)
    - 그 미만: 정상
    """
    model = load_model()

    # 이미지 불러오기
    files = [f for f in sorted(os.listdir(input_dir))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # 임베딩 추출
    embeddings = {}
    for f in files:
        path = os.path.join(input_dir, f)
        try:
            embeddings[f] = get_embedding(model, path)
        except Exception as e:
            print(f"⚠️ {f} 임베딩 추출 실패: {e}")

    results = []
    dup_groups = []

    # 모든 조합 비교
    for f1, f2 in itertools.combinations(files, 2):
        vec1, vec2 = embeddings[f1].reshape(1, -1), embeddings[f2].reshape(1, -1)
        sim = cosine_similarity(vec1, vec2)[0][0]

        if sim >= threshold:
            results.append([f1, f2, round(sim, 4), "중복", "다시 스캔 필요"])
            dup_groups.append((f1, f2, sim))
        elif sim >= suspect_low:
            results.append([f1, f2, round(sim, 4), "유사 후보", "검수 필요"])
        else:
            results.append([f1, f2, round(sim, 4), "다름", "-"])

    return results, dup_groups
