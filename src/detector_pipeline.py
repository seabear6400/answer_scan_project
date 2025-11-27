import os
import shutil
import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import zipfile
import re
import atexit
import threading

import numpy as np
from PIL import Image
import io
from PIL import UnidentifiedImageError
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

# 경량화 플래그 (주의)
# - 대부분의 선택적 외부 의존성 플래그는 실제 사용처가 없어 제거했습니다.
# - DINOv2(timm) 관련 체크용 `_HAS_TIMM`만 유지합니다. 필요 시 전역 구성으로 분리하세요.
_HAS_TIMM = False

from sklearn.neighbors import NearestNeighbors

# 모듈 로거 (조용한 모드)
import logging
logger = logging.getLogger(__name__)
# 로거 비활성화 - 토스트 창에서 진행상황을 보여주므로 콘솔 출력 숨김
logger.setLevel(logging.CRITICAL)  # CRITICAL만 표시 (거의 없음)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


_PRUNE_RESULTS_AFTER_ZIP = _env_flag("ANSWER_SCAN_PRUNE_RESULTS_AFTER_ZIP", True)
_PRUNE_IMMEDIATE = _env_flag("ANSWER_SCAN_PRUNE_RESULTS_IMMEDIATE", False)
_CLEANUP_LOCK = threading.Lock()
_CLEANUP_QUEUE: List[Path] = []
_CLEANUP_REGISTERED = False


def _flush_cleanup_queue() -> None:
    if not _PRUNE_RESULTS_AFTER_ZIP:
        return
    with _CLEANUP_LOCK:
        pending = list(_CLEANUP_QUEUE)
        _CLEANUP_QUEUE.clear()
    for target in pending:
        try:
            if target.exists():
                shutil.rmtree(target, onerror=_handle_remove_readonly)
        except Exception:
            warnings.warn(f"결과 폴더 정리 실패: {target}")


def _register_cleanup_hook() -> None:
    global _CLEANUP_REGISTERED
    if _CLEANUP_REGISTERED or not _PRUNE_RESULTS_AFTER_ZIP:
        return
    atexit.register(_flush_cleanup_queue)
    _CLEANUP_REGISTERED = True


def _schedule_result_dir_cleanup(path: Path) -> None:
    if not _PRUNE_RESULTS_AFTER_ZIP:
        return
    _register_cleanup_hook()
    with _CLEANUP_LOCK:
        if path not in _CLEANUP_QUEUE:
            _CLEANUP_QUEUE.append(path)
    if _PRUNE_IMMEDIATE:
        _flush_cleanup_queue()


def diagnose_gpu():
    """간단한 GPU 진단 유틸리티(출력 최소화).

    원래는 디버깅용으로 상세 출력을 했으나 소규모 배포에서는 콘솔 출력을 최소화합니다.
    이 함수는 `get_device_info()`와 동일한 정보를 조용히 반환합니다.
    """
    return get_device_info()


@dataclass
class DetectorConfig:
    # 백엔드 (성능 최적화를 위해 auto 우선)
    embed_backend: str = "auto"     # auto/dinov2/resnet18 - 자동 선택으로 최적화
    ann_backend: str = "auto"       # auto/brute/faiss/hnsw

    # GPU 사용 설정
    force_gpu: bool = False         # GPU를 강제로 시도 (False로 변경: 안정성 우선)
    fallback_to_cpu: bool = True    # GPU 실패 시 CPU로 fallback (항상 True)

    # ANN 파라미터
    k: int = 20
    hnsw_M: int = 32
    hnsw_efC: int = 200
    hnsw_efS: int = 64

    # 사전 필터 설정 (성능 최적화)
    prefilter: str = "phash"        # phash/pdq/both (phash가 더 빠름)
    phash_thresh: int = 12          # 약간 관대하게 설정
    pdq_thresh: int = 75            # 약간 엄격하게 설정  
    density_diff_thresh: float = 0.20  # 약간 관대하게 설정

    # 유사도 임계값 (성능 최적화를 위해 약간 관대하게)
    cnn_thresh: float = 0.98  # 약간 낮춰서 더 빠른 처리
    suspect_low: float = 0.93  # 의심 구간도 약간 낮춤

    # 공백(빈칸) 감지 (성능 최적화 + 정밀도 향상)
    blank_method: str = "sauvola"   # otsu/sauvola
    blank_density_thresh: float = 0.02  # 개선된 노이즈 필터링과 함께 빈칸 탐지 안정성 향상
    blank_border_trim: float = 0.02      # 공백 감지 시 가장자리 잘라내기 비율
    blank_min_component_ratio: float = 0.0008  # 노이즈 제거를 위한 최소 컴포넌트 비율
    blank_auto_tune: bool = True         # 데이터 기반 자동 임계값 조정
    blank_auto_suffix: str = "2"         # 자동 임계값을 적용할 파일명 접미사
    blank_auto_min_samples: int = 6      # 자동 임계값 계산에 필요한 최소 샘플 수
    blank_auto_margin: float = 0.005     # 자동 임계값에 적용할 안전 완충값 (기본 0.5%)
    blank_auto_cap: float = 0.12         # 자동 임계값 상한
    blank_binary_weight: float = 0.55    # 밀도 기반 점수 가중치
    blank_contrast_weight: float = 0.25  # 대비 기반 점수 가중치
    blank_edge_weight: float = 0.20      # 에지/텍스처 기반 점수 가중치
    blank_laplacian_ksize: int = 3       # 에지 추출 커널 크기 (홀수)

    # 재정렬 / OCR
    use_lpips: bool = False
    lpips_thresh: float = 0.2
    use_ocr: bool = False
    text_sim_thresh: float = 0.85

    # 정렬(Alignment) (현재 그룹핑에 사용되지 않음, main.py 호환용)
    use_alignment: bool = False

    # 임베딩 설정 (성능 최적화)
    batch_size: int = 128  # 더 큰 배치 크기로 처리량 향상
    num_workers: int = 4   # 멀티프로세싱 활성화
    roi_ratio: Tuple[float, float, float, float] = (0.15, 0.15, 0.85, 0.85)
    
    # 자동 최적화 설정
    auto_optimize: bool = True  # 데이터 크기에 따른 자동 최적화 활성화


# -------------------------- 적응적 최적화 ---------------------
def get_device_info():
    """GPU/CPU 디바이스 정보를 빠르게 조회합니다. GPU 실패 시 항상 CPU로 fallback."""
    device_info = {
        'has_gpu': False,
        'gpu_count': 0,
        'gpu_memory_gb': 0,
        'gpu_name': '',
        'device': torch.device('cpu')
    }
    
    try:
        # 빠른 CUDA 체크 (타임아웃 없이)
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            
            if gpu_count > 0:
                # 빠른 GPU 정보 수집
                try:
                    props = torch.cuda.get_device_properties(0)
                    device_info['has_gpu'] = True
                    device_info['gpu_count'] = gpu_count
                    device_info['gpu_memory_gb'] = props.total_memory / (1024**3)
                    device_info['gpu_name'] = props.name
                    device_info['device'] = torch.device('cuda:0')
                    
                    # GPU 감지 (조용히)
                    device_info['has_gpu'] = True
                    device_info['gpu_count'] = gpu_count
                    device_info['gpu_memory_gb'] = props.total_memory / (1024**3)
                    device_info['gpu_name'] = props.name
                    device_info['device'] = torch.device('cuda:0')
                    
                    # 빠른 GPU 테스트
                    try:
                        test_tensor = torch.zeros(2).cuda()
                        del test_tensor
                        torch.cuda.empty_cache()
                    except Exception:
                        device_info['has_gpu'] = False
                        device_info['device'] = torch.device('cpu')
                        
                except Exception as e:
                    logger.warning(f"GPU 초기화 실패: {e}")
                    device_info['has_gpu'] = False
                    device_info['device'] = torch.device('cpu')
            else:
                pass  # GPU 없음 - 조용히 CPU 사용
        else:
            pass  # CUDA 불가 - 조용히 CPU 사용
            
    except Exception as e:
        pass  # GPU 체크 실패 시 조용히 CPU 사용
    
    # 최종 디바이스만 간단히 표시
    if device_info['has_gpu']:
        logger.info(f"� GPU 모드")
    else:
        logger.info("� CPU 모드")
    
    return device_info


def optimize_config_for_data_size(cfg: DetectorConfig, n_images: int, device_info: dict = None) -> DetectorConfig:
    """
    데이터 크기 및 GPU 사양에 따라 설정을 적응적으로 최적화합니다.
    
    최적화 전략:
    - GPU 우선: GTX 1660 Ti 같은 GPU가 있으면 최대한 활용
    - 소규모 데이터(< 50): 고정 오버헤드 최소화
    - 중간 규모(50-500): 균형잡힌 설정  
    - 대규모 데이터(> 500): 배치 처리 최적화, 고성능 백엔드 활용
    """
    # 소규모(<=150) 전용: 복잡한 자동 튜닝을 제거하고 보수적인 고정값을 사용합니다.
    if not cfg.auto_optimize:
        return cfg

    optimized = DetectorConfig(**vars(cfg))

    # 간단하고 안전한 기본값들
    optimized.embed_backend = "resnet18"
    optimized.ann_backend = "brute"
    optimized.batch_size = min(32, max(8, n_images))
    optimized.num_workers = 0  # 안정성을 위해 기본은 단일 스레드
    optimized.use_lpips = False
    optimized.use_ocr = False

    # k는 데이터 크기를 넘지 않도록 보정
    optimized.k = min(int(getattr(cfg, 'k', 20)), max(1, n_images - 1))

    return optimized


# ---------------------- 성능 로깅 유틸리티 ----------------------
import csv
import platform
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


def _collect_run_features(input_paths: List[str], cfg: DetectorConfig, times: Dict[str, float]) -> Dict:
    """실행에 대한 간단한 특성(피처)을 수집하여 성능 로깅에 사용합니다.

    CSV에 바로 추가할 수 있는 평탄한 딕셔너리를 반환합니다.
    """
    sizes = []
    widths = []
    heights = []
    for p in input_paths:
        try:
            sizes.append(os.path.getsize(p))
        except Exception:
            sizes.append(0)
        try:
            with Image.open(p) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except Exception:
            widths.append(0)
            heights.append(0)
    
    n = len(input_paths)
    
    # GPU 정보 상세 수집
    device_info = get_device_info()
    
    feat = {
        "timestamp": int(time.time()),
        "platform": platform.system(),
        "n_images": n,
        "mean_size": float(np.mean(sizes)) if n else 0.0,
        "mean_w": float(np.mean(widths)) if n else 0.0,
        "mean_h": float(np.mean(heights)) if n else 0.0,
        "batch_size": int(getattr(cfg, "batch_size", 0)),
        "num_workers": int(getattr(cfg, "num_workers", 0)),
        "use_ocr": int(bool(getattr(cfg, "use_ocr", False))),
        "use_lpips": int(bool(getattr(cfg, "use_lpips", False))),
        "embed_backend": str(getattr(cfg, "embed_backend", "")),
        "gpu": int(device_info['has_gpu']),
        "gpu_count": int(device_info['gpu_count']),
        "gpu_memory_gb": round(device_info['gpu_memory_gb'], 1),
        "gpu_name": str(device_info['gpu_name']),
    }
    try:
        if _HAS_PSUTIL:
            vm = psutil.virtual_memory()
            feat.update({"mem_total": int(vm.total), "cpu_count": int(psutil.cpu_count(logical=True))})
        else:
            feat.update({"mem_total": 0, "cpu_count": int(os.cpu_count() or 0)})
    except Exception:
        feat.update({"mem_total": 0, "cpu_count": int(os.cpu_count() or 0)})
    # 측정된 시간들을 추가
    feat.update(times)
    return feat


def _append_perf_csv(output_dir: str, row: Dict):
    art = os.path.join(output_dir, "artifacts")
    os.makedirs(art, exist_ok=True)
    csvf = os.path.join(art, "perf_runs.csv")
    header = list(row.keys())
    write_header = not os.path.exists(csvf)
    try:
        with open(csvf, "a", newline="", encoding="utf-8") as fw:
            writer = csv.DictWriter(fw, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        logger.warning("perf_runs.csv 기록 실패")


# -------------------------- ROI / 헬퍼 --------------------------
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


# -------------------------- 사전 필터 ------------------------------
def phash_of(path: str, roi_ratio: Tuple[float, float, float, float]) -> Optional[imagehash.ImageHash]:
    try:
        img = Image.open(path)
        img = crop_roi(img, roi_ratio)
        return imagehash.phash(img)
    except Exception as e:
        logger.warning(f"phash 계산 실패: {path} -> {e}")
        return None


# PDQ 관련 코드는 소규모(<=150) 전용 구성에서 사용하지 않습니다.
# 필요하면 이 주석을 제거하고 pdqhash 관련 코드를 복원하세요.


# PDQ 관련 계산은 소규모 전용에서 사용하지 않으므로 관련 헬퍼를 제거했습니다.


def ink_density(
    path: str,
    roi_ratio: Tuple[float, float, float, float],
    method: str = "sauvola",
    border_trim: float = 0.0,
    min_component_ratio: float = 0.0,
    binary_weight: float = 0.7,
    contrast_weight: float = 0.3,
    edge_weight: float = 0.0,
    laplacian_ksize: int = 3,
) -> float:
    gray = read_gray(path)
    if gray is None:
        logger.warning(f"ink_density: 이미지 로드 실패로 0 반환: {path}")
        return 0.0
    h, w = gray.shape[:2]
    l, t, r, b = roi_ratio
    x1, y1, x2, y2 = int(l * w), int(t * h), int(r * w), int(b * h)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    if border_trim > 0.0:
        trim_x = int(border_trim * roi.shape[1])
        trim_y = int(border_trim * roi.shape[0])
        if trim_x * 2 < roi.shape[1] and trim_y * 2 < roi.shape[0]:
            roi = roi[trim_y:roi.shape[0] - trim_y, trim_x:roi.shape[1] - trim_x]
        if roi.size == 0:
            return 0.0

    if min(roi.shape[:2]) >= 5:
        roi_proc = cv2.GaussianBlur(roi, (5, 5), 0)
    else:
        roi_proc = roi.copy()

    # 조명 보정 및 대비 향상 (빈칸 노이즈 제거용)
    if roi_proc.size and roi_proc.max() > roi_proc.min():
        roi_proc = cv2.normalize(roi_proc, None, 0, 255, cv2.NORM_MINMAX)

    # 소규모 전용 간소화: Sauvola(추가 의존성) 대신 항상 Otsu 임계값을 사용합니다.
    # 이유: 작은 배치(<=150)에서는 Otsu가 충분히 안정적이며 외부 의존성을 줄여 설치/실행을 단순화합니다.
    _, thr = cv2.threshold(roi_proc, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = (thr > 0).astype(np.uint8)

    if min(binary.shape) >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    dominant_component_ratio = 0.0
    if min_component_ratio > 0.0 and binary.size:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            min_area = max(4, int(binary.size * min_component_ratio))
            filtered = np.zeros_like(binary, dtype=np.uint8)
            keep_area = 0
            for lbl in range(1, num_labels):
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area >= min_area:
                    filtered[labels == lbl] = 1
                    keep_area += area
            binary = filtered
            if keep_area > 0:
                dominant_component_ratio = float(keep_area) / float(binary.size)

    if binary.size == 0:
        return 0.0

    binary_density = float(np.count_nonzero(binary)) / float(binary.size)
    mean_dark = max(0.0, 1.0 - float(np.mean(roi_proc)) / 255.0)
    std_dark = float(np.std(roi_proc)) / 255.0
    contrast_score = max(0.0, min(1.0, 0.5 * mean_dark + 0.5 * std_dark))

    edge_score = 0.0
    if edge_weight > 0.0 and min(roi_proc.shape[:2]) >= 5:
        ksize = laplacian_ksize if laplacian_ksize % 2 == 1 else 3
        lap = cv2.Laplacian(roi_proc, cv2.CV_32F, ksize=ksize)
        lap_abs = np.abs(lap)
        lap_abs = np.clip(lap_abs, 0.0, 255.0)
        edge_score = float(np.mean(lap_abs)) / 255.0
        # 큰 컴포넌트가 존재하면 에지 점수에 가중치 부여 (필기 강조)
        if dominant_component_ratio > 0.0:
            edge_score = min(1.0, edge_score + dominant_component_ratio)

    # 가중 조합 (합이 0이면 기본값 사용)
    bw = max(0.0, binary_weight)
    cw = max(0.0, contrast_weight)
    ew = max(0.0, edge_weight)
    weight_sum = bw + cw + ew
    if weight_sum <= 0.0:
        bw, cw = 0.7, 0.3
        weight_sum = bw + cw
    density = (bw * binary_density + cw * contrast_score + ew * edge_score) / weight_sum
    return float(max(0.0, min(1.0, density)))


# -------------------------- 데이터셋 / 임베딩 ----------------------
class ImgDataset(Dataset):
    def __init__(self, paths: List[str], roi_ratio: Tuple[float, float, float, float], backend: str):
        self.paths = paths
        self.roi = roi_ratio
        self.backend = backend
        
        # DINOv2 입력 크기 문제 해결
        if backend == "dinov2":
            self.tf = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),  # DINOv2는 정확히 224x224 필요
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet 정규화
            ])
        else:  # ResNet18
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
            # 바이트 기반 로딩으로 경로 문제 해결
            with open(p, 'rb') as f:
                data = f.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            # 폴백: 직접 로딩
            img = Image.open(p).convert("RGB")
        
        # ROI 크롭
        img = crop_roi(img, self.roi)
        
        # 변환 적용
        tensor = self.tf(img)
        
        # 디버깅: 텐서 크기 확인
        if tensor.shape != (3, 224, 224):
            logger.warning(f"잘못된 텐서 크기: {tensor.shape}, 경로: {p}")
            # 강제로 224x224로 리사이즈
            tensor = transforms.Resize((224, 224))(tensor)
        
        return tensor, p


def load_model(device: torch.device, backend: str, force_gpu: bool = False, fallback_to_cpu: bool = True) -> nn.Module:
    """
    안정적인 모델 로딩: GPU 실패 시 항상 CPU로 fallback
    DINOv2 입력 크기 문제 해결
    """
    # 소규모 전용 간소화: ResNet18 만 사용합니다. (DINOv2 등 무거운 모델 제거)
    # GPU가 사용 가능하면 그 장치로, 아니면 CPU로 모델을 로드합니다.
    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
        model = model.eval().to(device)
        return model
    except Exception as e:
        # 단순한 폴백: CPU에서 시도
        device_cpu = torch.device('cpu')
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
        model = model.eval().to(device_cpu)
        return model


def compute_embeddings(paths: List[str], device: torch.device, batch_size: int, num_workers: int,
                       roi_ratio: Tuple[float, float, float, float], backend: str,
                       progress_callback: Optional[callable] = None, force_gpu: bool = False):
    """
    안정적인 임베딩 계산: GPU 실패 시 자동으로 CPU fallback
    입력 크기 검증 추가
    """
    # GPU 강제 사용 체크
    if force_gpu and torch.cuda.is_available() and device.type == "cpu":
        logger.info("💡 GPU 강제 사용: CPU → GPU 시도")
        device = torch.device("cuda:0")
        try:
            torch.cuda.empty_cache()
        except:
            logger.warning("GPU 메모리 정리 실패, 계속 진행")
    
    ds = ImgDataset(paths, roi_ratio, backend)
    
    # GPU 사용 시 pin_memory 최적화
    pin_memory = False
    try:
        pin_memory = (device.type == "cuda" and torch.cuda.is_available())
    except:
        pin_memory = False
    
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    t_model0 = time.time()
    
    try:
        model = load_model(device, backend, force_gpu, fallback_to_cpu=True)
        actual_device = next(model.parameters()).device
        if actual_device != device:
            device = actual_device
    except Exception as e:
        device = torch.device("cpu")
        model = load_model(device, "resnet18", force_gpu=False, fallback_to_cpu=True)
    
    t_model1 = time.time()
    model_load_s = float(t_model1 - t_model0)

    # GPU 메모리 정보 로깅 (간소화)
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    embs = []
    ordered_paths = []
    total = len(ds)
    processed = 0
    
    with torch.no_grad():
        for batch_idx, (x, pths) in enumerate(dl):
            try:
                # 입력 크기 검증
                if x.shape[1:] != (3, 224, 224):
                    logger.warning(f"배치 {batch_idx}: 잘못된 입력 크기 {x.shape}")
                    # 크기 강제 조정
                    x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                
                # GPU로 데이터 이동
                if device.type == "cuda":
                    x = x.to(device, non_blocking=True)
                else:
                    x = x.to(device)
                
                # 모델 추론
                out_t = model(x)
                out = out_t.detach().cpu().numpy()
                
                # 출력 형태 정규화
                if out.ndim > 2:
                    out = out.reshape(out.shape[0], -1)
                out = out.astype(np.float32)
                
                embs.append(out)
                ordered_paths.extend(list(pths))
                processed += out.shape[0]
                
                # 진행률 콜백
                if progress_callback is not None and total > 0:
                    try:
                        progress_callback('embed', float(processed) / float(total), f"{processed}/{total}")
                    except Exception:
                        pass
                        
            except Exception as e:
                logger.error(f"임베딩 배치 {batch_idx} 처리 실패: {e}")
                # 심각한 오류면 CPU로 재시도
                if device.type == "cuda" and "doesn't match model" in str(e):
                    logger.error("모델 입력 크기 불일치 - CPU로 재시도")
                    raise e  # 상위에서 CPU 재시도하도록
                continue
    
    # 결과 정리
    if len(embs):
        embs = np.vstack(embs)
    else:
        # 빈 결과에 대한 차원 추정
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

    # 일관성 검사
    if len(ordered_paths) != len(paths):
        raise RuntimeError(f"compute_embeddings: ordered_paths ({len(ordered_paths)}) != input paths ({len(paths)}) — 일부 이미지 처리가 실패했습니다.")

    # 최종 콜백
    if progress_callback is not None:
        try:
            progress_callback('embed', 1.0, f"{len(ordered_paths)}/{total}")
        except Exception:
            pass

    # GPU 메모리 정리
    try:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    return embs, ordered_paths, model_load_s


# -------------------------- ANN 구성 -----------------------------
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
        반환값은 (indices, sims, backend_used) 형태입니다.
    """
    N, D = embs.shape
    if N == 0:
        return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=np.float32), "none"

    # 소규모(<=150) 전용: brute-force(코사인)만 사용하여 복잡한 외부 인덱스 의존성을 제거합니다.
    mat = embs.astype(np.float32)
    if mat.size:
        mat = l2_normalize(mat)

    nn = NearestNeighbors(n_neighbors=min(k + 1, N), metric="cosine", algorithm="brute")
    nn.fit(mat)
    dists, idxs = nn.kneighbors(mat, return_distance=True)
    sims = 1.0 - dists
    return idxs, sims, "brute"


# LPIPS / OCR / RapidFuzz / token_set_ratio 등 무거운 선택적 기능은 소규모 전용에서는 사용하지 않습니다.
# 관련 함수들이 필요하면 별도로 추가하세요.


# -------------------------- 유틸 -------------------
def _handle_remove_readonly(func, path, exc_info):
    # 읽기 전용 파일도 강제 삭제
    os.chmod(path, stat.S_IWRITE)
    func(path)

def _copy_to_dir(src: str, dst_dir: str):
    """원본 파일(src)을 파일 이름은 그대로 유지한 채 대상 디렉터리(dst_dir)로 복사합니다. 대상 디렉터리가 없으면 생성합니다."""
    try:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
        return True
    except Exception as e:
        logger.warning(f"복사 실패 {src} -> {dst_dir}: {e}")
        return False


def _safe_recreate_dir(path: str, retries: int = 3, delay: float = 0.5):
    """
    디렉터리(폴더)를 삭제하고 새로 만들 때 발생하는 Windows 파일 잠금이나 일시적인 권한 문제에 대비하여,
    여러 번 재시도하는 로직을 적용합니다.
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


def _write_zip_status(status_dir: Path, status_name: str, zip_path: Path, candidates: Sequence[Path]) -> None:
    """ZIP 생성 이력을 남겨 운영자가 확인할 수 있도록 상태 파일을 기록합니다."""
    try:
        status_dir.mkdir(parents=True, exist_ok=True)
        status_path = status_dir / f"{status_name}_status.txt"
        lines = [
            f"zip_created: {zip_path}",
            f"candidates: {[str(p) for p in candidates]}",
        ]
        status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        logger.warning("ZIP 상태 파일 기록 실패")


def _zippable_files(root: Path) -> Iterable[Path]:
    """ZIP에 포함할 파일 목록을 생성합니다."""
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _verify_zip_integrity(zip_path: Path) -> bool:
    """ZIP 파일 무결성을 빠르게 검증합니다."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            corrupted = zf.testzip()
            if corrupted:
                warnings.warn(f"ZIP 무결성 검사 실패: {corrupted}")
                return False
    except Exception as exc:
        warnings.warn(f"ZIP 무결성 검사 중 오류 발생: {exc}")
        return False
    return True


def create_result_zip_for_dir(result_dir: str, zip_basename: Optional[str] = None) -> Optional[str]:
    """단일 결과 폴더 전체를 ZIP 아카이브로 생성합니다.

    동작 요약 (한국어):
    - ZIP 파일은 결과 폴더(`result_dir`)의 부모 디렉터리(=결과 폴더와 동일 레벨)에 생성됩니다.
      예: `/some/path/11001_결과` -> `/some/path/11001_결과_163... .zip`
    - ZIP 생성 이력 기록은 생략하여 불필요한 폴더 생성을 방지합니다.
    """
    try:
        target = Path(result_dir).resolve()
    except Exception:
        target = Path(result_dir)

    if not target.exists() or not target.is_dir():
        return None

    # ZIP 생성 시 불필요한 artifacts 폴더 생성 방지: 상태 파일 기록 제거
    # artifacts_dir = target / "artifacts"
    # try:
    #     artifacts_dir.mkdir(parents=True, exist_ok=True)
    # except Exception:
    #     # artifacts 생성 실패 시에는 상태 기록이 불가하므로 중단
    #     return None
 
    base_name = zip_basename or target.name

    # ZIP은 결과 폴더의 부모 디렉터리에 생성 (요구사항: 결과 폴더와 동일 레벨)
    zip_parent = target.parent
    try:
        zip_parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    # 이미 동일 베이스명으로 부모 디렉터리에 생성된 ZIP이 있으면 재사용
    try:
        existing = sorted(zip_parent.glob(f"{base_name}_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if existing:
            selected = existing[0]
            _schedule_result_dir_cleanup(target)
            return str(selected)
    except Exception:
        pass

    timestamp = int(time.time())
    zip_path = zip_parent / f"{base_name}_{timestamp}.zip"

    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in _zippable_files(target):
                # ZIP 파일 자체가 대상 경로에 우연히 포함되는 경우(희박) 스킵
                if file_path.resolve() == zip_path.resolve():
                    continue
                try:
                    arcname = file_path.relative_to(target)
                except ValueError:
                    arcname = file_path.name
                zf.write(file_path, arcname)
    except Exception:
        logger.warning("결과 ZIP 생성 실패", exc_info=True)
        try:
            if zip_path.exists():
                zip_path.unlink()
        except Exception:
            pass
        return None

    if not _verify_zip_integrity(zip_path):
        try:
            zip_path.unlink(missing_ok=True)
        except TypeError:
            # Python 3.8 호환: missing_ok 인자 미지원
            try:
                if zip_path.exists():
                    zip_path.unlink()
            except Exception:
                pass
        return None

    _schedule_result_dir_cleanup(target)

    # ZIP 생성 시 불필요한 artifacts 폴더 생성 방지: 상태 파일 기록 제거
    # _write_zip_status(artifacts_dir, base_name, zip_path, [target])
    return str(zip_path)


def create_aggregate_result_zip(base_dir: str, target_dir: Optional[str] = None, prefix: Optional[str] = None) -> Optional[str]:
    """여러 결과 폴더를 하나의 압축 파일로 통합해 총괄 ZIP을 생성합니다.

    한국어 요약:
    - `base_dir` 아래에서 `*_결과` 패턴을 만족하는 폴더들을 찾아 하나의 묶음으로 압축합니다.
    - `target_dir`를 지정하면 해당 경로(예: 상위 폴더)에 ZIP을 생성하며, 지정하지 않으면 `base_dir`에 생성합니다.
    - `prefix`를 전달하면 ZIP 파일명 및 내부 루트 폴더명에 접두사로 사용합니다.
      예: prefix='인문계' → `인문계_총_결과_<timestamp>.zip`
    - ZIP 생성 이력 기록은 생략하여 불필요한 폴더 생성을 방지합니다.
    """

    try:
        base_path = Path(base_dir).resolve()
    except Exception:
        base_path = Path(base_dir)

    if not base_path.exists() or not base_path.is_dir():
        return None

    # 1) 묶을 결과 폴더 탐색: 기본 정책은 `숫자_결과(_번호)` 패턴만 포함
    result_pattern = re.compile(r"^\d+_결과(?:_\d+)?$")
    result_dirs: List[Path] = []
    seen: set[str] = set()

    if result_pattern.match(base_path.name):
        key = str(base_path)
        seen.add(key)
        result_dirs.append(base_path)

    try:
        for root, dirnames, _ in os.walk(base_path):
            for dirname in dirnames:
                if not result_pattern.match(dirname):
                    continue
                candidate = Path(root) / dirname
                try:
                    resolved = candidate.resolve()
                except Exception:
                    resolved = candidate
                key = str(resolved)
                if key in seen:
                    continue
                seen.add(key)
                result_dirs.append(resolved)
    except Exception:
        return None

    if not result_dirs:
        return None

    # 2) ZIP 생성 위치 결정 (예: 인문계를 분석하면 상위 폴더에 생성)
    if target_dir:
        try:
            target_root = Path(target_dir).resolve()
        except Exception:
            target_root = Path(target_dir)
    else:
        target_root = base_path

    try:
        target_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    bundle_prefix = (prefix or base_path.name or "aggregate").strip()
    if not bundle_prefix:
        bundle_prefix = "aggregate"
    zip_base_name = f"{bundle_prefix}_총_결과"

    # 3) 기존 ZIP이 최신이라면 재사용 (결과 폴더보다 새로우면 그대로 반환)
    latest_source_mtime = 0.0
    try:
        latest_source_mtime = max(
            os.path.getmtime(str(p))
            for p in result_dirs
            if p.exists()
        )
    except Exception:
        latest_source_mtime = 0.0

    try:
        existing = sorted(
            target_root.glob(f"{zip_base_name}_*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if existing:
            latest_zip = existing[0]
            try:
                if latest_zip.stat().st_mtime >= latest_source_mtime:
                    return str(latest_zip)
            except Exception:
                pass
    except Exception:
        pass

    # 4) 새 ZIP 생성
    timestamp = int(time.time())
    bundle_dir_name = f"{zip_base_name}_{timestamp}"
    zip_path = target_root / f"{bundle_dir_name}.zip"

    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for res_dir in result_dirs:
                rel_root = Path(bundle_dir_name) / res_dir.name
                try:
                    for file_path in _zippable_files(res_dir):
                        try:
                            arcname = rel_root / file_path.relative_to(res_dir)
                        except ValueError:
                            arcname = rel_root / file_path.name
                        zf.write(file_path, arcname.as_posix())
                except Exception:
                    logger.warning("총괄 ZIP 생성 중 일부 폴더를 건너뜀: %s", res_dir, exc_info=True)
                    continue
    except Exception:
        try:
            if zip_path.exists():
                zip_path.unlink()
        except Exception:
            pass
        return None

    # ZIP 생성 시 불필요한 artifacts 폴더 생성 방지: 상태 파일 기록 제거
    # status_dir = base_path / "artifacts"
    # try:
    #     _write_zip_status(status_dir, zip_base_name, zip_path, result_dirs)
    # except Exception:
    #     # 상태 파일 기록 실패는 치명적이지 않으므로 무시
    #     pass

    return str(zip_path)


# ---------------------- 아티팩트 / 병렬 헬퍼 ----------------------
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
    # IO 중심의 메타데이터 작업을 위해 ThreadPoolExecutor에서 실행되는 워커
    f, p, cfg = args
    ph = None
    dens = 0.0
    try:
        if cfg.prefilter in ("phash", "both"):
            ph = phash_of(p, cfg.roi_ratio)
    except Exception:
        ph = imagehash.hex_to_hash("0" * 16)
    try:
        dens = ink_density(p, cfg.roi_ratio, cfg.blank_method)
    except Exception:
        dens = 0.0
    # 간소화: PDQ/OCR 계산은 소규모 전용에서 생략합니다. (정확도 핵심: pHash + density 유지)
    return f, ph, None, float(dens), ""


def _auto_blank_threshold(values: List[float], cfg: DetectorConfig) -> Optional[float]:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float32)
    if arr.size < cfg.blank_auto_min_samples:
        return None
    arr = arr[(arr >= 0.0) & (arr <= 1.0)]
    if arr.size < cfg.blank_auto_min_samples:
        return None

    c1 = float(np.percentile(arr, 25))
    c2 = float(np.percentile(arr, 75))
    if np.isclose(c1, c2, atol=1e-4):
        return None

    for _ in range(12):
        dist1 = np.abs(arr - c1)
        dist2 = np.abs(arr - c2)
        assign = dist1 <= dist2
        if assign.all() or (~assign).all():
            return None
        new_c1 = float(arr[assign].mean()) if assign.any() else c1
        new_c2 = float(arr[~assign].mean()) if (~assign).any() else c2
        if np.isnan(new_c1) or np.isnan(new_c2):
            return None
        if abs(new_c1 - c1) < 1e-5 and abs(new_c2 - c2) < 1e-5:
            c1, c2 = new_c1, new_c2
            break
        c1, c2 = new_c1, new_c2

    if c1 > c2:
        c1, c2 = c2, c1
    # 기본적으로 두 클러스터의 중간값을 반환합니다.
    # 실제 판정에서는 별도의 안전 완충(blank_auto_margin)을 적용하여
    # 임계 바로 아래의 경계 케이스에서 오탐을 줄입니다.
    threshold = float((c1 + c2) / 2.0)
    threshold = max(cfg.blank_density_thresh, threshold)
    threshold = min(cfg.blank_auto_cap, max(0.0, threshold))
    if threshold <= cfg.blank_density_thresh + 1e-6:
        return None
    return threshold


def _build_blank_flags(files: List[str], densities: Dict[str, float], cfg: DetectorConfig) -> Tuple[Dict[str, bool], Dict[str, float], Optional[float]]:
    thresholds: Dict[str, float] = {f: cfg.blank_density_thresh for f in files}
    auto_threshold = None
    target_suffix = cfg.blank_auto_suffix
    if cfg.blank_auto_tune and target_suffix:
        suffix_vals = [densities.get(f, 0.0) for f in files if os.path.splitext(f)[0].endswith(target_suffix)]
        auto_threshold = _auto_blank_threshold(suffix_vals, cfg)
        if auto_threshold is not None:
            for f in files:
                if os.path.splitext(f)[0].endswith(target_suffix):
                    thresholds[f] = max(thresholds[f], auto_threshold)

    # 적용할 안전 여유(margin)를 가져옵니다.
    margin = float(getattr(cfg, "blank_auto_margin", 0.0))

    blank_flags: Dict[str, bool] = {}
    # thresholds 딕셔너리에 실제 판정에 사용된 유효 임계값(effective threshold)을 기록합니다.
    for f in files:
        d = float(densities.get(f, 0.0))
        t = float(thresholds.get(f, cfg.blank_density_thresh))
        # eff_t는 auto에서 계산된 t에서 margin을 빼고, 전역 최소 임계값을 넘지 않도록 보호합니다.
        eff_t = max(t - margin, float(cfg.blank_density_thresh))
        thresholds[f] = eff_t
        blank_flags[f] = d < eff_t

    return blank_flags, thresholds, auto_threshold


def _pair_and_group(name_by_row: Dict[int, str], idxs: np.ndarray, sims: np.ndarray,
                    phashes: Dict[str, imagehash.ImageHash], pdqs: Dict[str, Optional[np.ndarray]],
                    densities: Dict[str, float], blank_flags: Dict[str, bool], texts: Dict[str, str], cfg: DetectorConfig,
                    input_dir: Optional[str] = None, path_map: Optional[Dict[str, str]] = None):
    """
    공통의 페어링/그룹화 로직을 추출한 헬퍼.
    name_by_row: 행 인덱스 -> 파일명 매핑
    input_dir OR path_map 중 하나를 제공하여 추가 검사(LPIPS/OCR)를 수행함.
    blank_flags: 사전에 계산된 공백 여부 (True이면 그룹 후보에서 제거).
    반환: pair_rows, groups
    """
    n = len(name_by_row)

    def _get_path(fname: str) -> str:
        if path_map is not None:
            return path_map.get(fname, fname)
        if input_dir is not None:
            return os.path.join(input_dir, fname)
        return fname

    def prefilter_ok(fi: str, fj: str) -> bool:
        if cfg.prefilter in ("phash", "both"):
            if abs(phashes.get(fi, imagehash.hex_to_hash("0"*16)) - phashes.get(fj, imagehash.hex_to_hash("0"*16))) > cfg.phash_thresh:
                return False
        # PDQ 관련 필터는 소규모 구성에서 비활성화됨 (pdq 연산은 사용하지 않음)
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

            # 파일명 끝이 '2'인 쌍만 그룹화 대상
            if not (os.path.splitext(fi)[0].endswith('2') and os.path.splitext(fj)[0].endswith('2')):
                continue

            if blank_flags.get(fi, False) or blank_flags.get(fj, False):
                continue

            if not prefilter_ok(fi, fj):
                continue

            sim = float(sims[i, col])
            all_pair_records.append((fi, fj, sim))

            # 단순화: 소규모 전용에서는 CNN 유사도 기준만으로 확정 판정
            confirmed = sim >= cfg.cnn_thresh

            if confirmed:
                confirmed_edges.append((i, j, sim))

    groups: Dict[str, List[str]] = {}
    gid_counter = 1
    if confirmed_edges:
        parents: Dict[int, int] = {}

        def find(x: int) -> int:
            while parents[x] != x:
                parents[x] = parents[parents[x]]
                x = parents[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            parents[rb] = ra

        nodes = set()
        for u, v, _w in confirmed_edges:
            nodes.add(u); nodes.add(v)
        for node in nodes:
            parents[node] = node
        for u, v, _w in confirmed_edges:
            union(u, v)

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

    # pair_rows 생성
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

    return pair_rows, groups


def _save_reports_and_copy(output_dir: str, files: List[str], densities: Dict[str, float], cfg: DetectorConfig,
                           blank_flags: Dict[str, bool], blank_thresholds: Dict[str, float],
                           pair_rows: List[List], groups: Dict[str, List[str]],
                           input_dir: Optional[str] = None, path_map: Optional[Dict[str, str]] = None,
                           embs: Optional[np.ndarray] = None, backend_used: Optional[str] = None,
                           auto_blank_threshold: Optional[float] = None):
    """
    공통 리포트 저장 및 파일 복사 로직.
    input_dir이 주어지면 detect_pipeline 스타일 동작(빈칸 기본 복사),
    path_map이 주어지면 detect_pipeline_files 스타일 동작(빈칸 복사 조건이 다름).
    """
    csv_path = os.path.join(output_dir, "report.csv")
    parquet_path = os.path.join(output_dir, "report.parquet")
    # 페어 행 데이터프레임 준비
    df_pairs = pd.DataFrame(pair_rows, columns=["파일1", "파일2", "유사도", "상태", "그룹ID"])

    # 변경된 동작: 그룹화가 전혀 발생하지 않은 경우(groups가 비어있는 경우)
    # report.csv / report.parquet은 '완전한 빈 리포트(헤더만)'로 처리합니다.
    # 이유: 그룹화가 이루어지지 않았는데도 유사 후보 페어 등 임의의 데이터가
    # 들어가면 이후 UI나 후처리에서 잘못 해석될 수 있으므로 명확히 구분하기 위함입니다.
    try:
        if not groups:
            # 그룹이 비어있을 때는 헤더만 있는 빈 DataFrame을 저장
            empty_df = pd.DataFrame(columns=["파일1", "파일2", "유사도", "상태", "그룹ID"])
            empty_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            try:
                # Parquet도 동일하게 빈 테이블로 저장 시도
                pl.from_pandas(empty_df).write_parquet(parquet_path)
            except Exception:
                # Parquet 쓰기 실패는 경고만 남기고 진행
                logger.warning("Parquet 생성 실패(그룹 없음)")
        else:
            # 정상적으로 그룹이 존재하면 모든 페어를 저장
            df_pairs.to_csv(csv_path, index=False, encoding="utf-8-sig")
            pl.from_pandas(df_pairs).write_parquet(parquet_path)
    except Exception:
        logger.warning("리포트 저장 실패")

    # 이미지 요약
    try:
        auto_cols = []
        for f in files:
            if auto_blank_threshold is not None and os.path.splitext(f)[0].endswith(cfg.blank_auto_suffix):
                auto_cols.append(auto_blank_threshold)
            else:
                auto_cols.append(np.nan)
        img_df = pd.DataFrame({
            "파일": files,
            "밀도": [densities.get(f, 0.0) for f in files],
            "적용임계값": [blank_thresholds.get(f, cfg.blank_density_thresh) for f in files],
            "자동보정임계값": auto_cols,
            "빈칸여부": [bool(blank_flags.get(f, False)) for f in files],
        })
        img_df.to_csv(os.path.join(output_dir, "images_summary.csv"), index=False, encoding="utf-8-sig")
    except Exception:
        logger.warning("images_summary 저장 실패")

    # grouped 복사
    for gid, members in groups.items():
        gdir = os.path.join(output_dir, "grouped", gid)
        for m in members:
            src = path_map[m] if path_map is not None else os.path.join(input_dir or "", m)
            _copy_to_dir(src, gdir)

    okdir = os.path.join(output_dir, "ok")
    bdir = os.path.join(output_dir, "blank_answers")
    os.makedirs(okdir, exist_ok=True)
    os.makedirs(bdir, exist_ok=True)

    grouped_set = set(itertools.chain.from_iterable(groups.values())) if groups else set()
    # 디버그: 빈칸 판단 로그를 artifacts에 남김
    try:
        dbg_dir = os.path.join(output_dir, "artifacts")
        os.makedirs(dbg_dir, exist_ok=True)
        dbg_csv = os.path.join(dbg_dir, "blank_debug.csv")
        with open(dbg_csv, "w", encoding="utf-8") as fdbg:
            fdbg.write("파일,밀도,임계값,빈칸여부\n")
            for f in files:
                dens = densities.get(f, 0.0)
                th = blank_thresholds.get(f, cfg.blank_density_thresh)
                flag = bool(blank_flags.get(f, dens < th))
                fdbg.write(f"{f},{dens:.6f},{th:.6f},{int(flag)}\n")
    except Exception:
        pass

    for f in files:
        src = path_map[f] if path_map is not None else os.path.join(input_dir or "", f)
        name_wo_ext = os.path.splitext(f)[0]
        # 변경: 일관성 있게 '<' 기준으로 판단
        is_blank = bool(blank_flags.get(f, densities.get(f, 0.0) < blank_thresholds.get(f, cfg.blank_density_thresh)))

        if path_map is None:
            # detect_pipeline 동작: 기본적으로 blank는 blank_answers로, 단 파일명 끝이 '1'이면 ok로 재분류
            if is_blank:
                if name_wo_ext.endswith('1'):
                    _copy_to_dir(src, okdir)
                else:
                    _copy_to_dir(src, bdir)
            elif f not in grouped_set:
                _copy_to_dir(src, okdir)
        else:
            # detect_pipeline_files 동작: blank는 '*2'로 끝나는 것만 bdir로 복사, '1'은 ok
            if is_blank:
                if name_wo_ext.endswith('2'):
                    _copy_to_dir(src, bdir)
                elif name_wo_ext.endswith('1'):
                    _copy_to_dir(src, okdir)
                else:
                    pass
            elif f not in grouped_set:
                _copy_to_dir(src, okdir)

    # 아티팩트 저장
    try:
        if embs is not None:
            np.save(os.path.join(output_dir, "artifacts", "embeddings.npy"), embs)
        if backend_used is not None:
            with open(os.path.join(output_dir, "artifacts", "ann_backend.txt"), "w", encoding="utf-8") as fw:
                fw.write(backend_used)
        with open(os.path.join(output_dir, "artifacts", "blank_threshold.txt"), "w", encoding="utf-8") as fw:
            fw.write(f"base_threshold={cfg.blank_density_thresh}\n")
            if auto_blank_threshold is not None:
                fw.write(f"auto_threshold={auto_blank_threshold}\n")
                fw.write(f"auto_suffix={cfg.blank_auto_suffix}\n")
    except Exception:
        pass


# -------------------------- 메인 파이프라인 ------------------------------
def detect_pipeline(input_dir: str, output_dir: str,
                    config: Optional[DetectorConfig] = None,
                    filter_func: Optional[callable] = None,
                    recursive: bool = False,
                    progress_callback: Optional[callable] = None,
                    **_deprecated_kwargs):
    cfg = config or DetectorConfig()

    def _cb(stage: str, pct: float = 0.0, msg: str = ""):
        # 안전하게 progress callback 호출
        try:
            if callable(progress_callback):
                progress_callback(stage=stage, pct=float(pct), msg=str(msg))
        except Exception:
            pass

    _cb("init", 0.01, "출력 폴더 초기화 중")
    # ✅ output 폴더 전체를 완전히 삭제 후 재생성 (모든 하위 폴더/파일 초기화)
    ok = _safe_recreate_dir(output_dir, retries=5, delay=0.5)
    if not ok:
        # 재시도에도 실패하면 명확한 에러를 던집니다.
        raise RuntimeError(f"Failed to initialize output dir: {output_dir}")
    # 하위 폴더도 재생성
    for sub in ["grouped", "ok", "blank_answers", "artifacts"]:
        subp = os.path.join(output_dir, sub)
        if not _safe_recreate_dir(subp, retries=3, delay=0.2):
            warnings.warn(f"하위 디렉터리(서브 디렉터리/하위 폴더) 생성이 실패했음 계속 진행 : {subp}")
    if not _safe_recreate_dir(os.path.join(output_dir, "artifacts", "thumbnails"), retries=3, delay=0.2):
        warnings.warn("썸네일 디렉터리 생성에 실패했으나 계속 진행합니다.")
    _cb("init", 0.03, "하위 폴더 생성 완료")

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

    # 데이터 크기 및 GPU에 따른 설정 최적화 (안전한 방식)
    n_images = len(files)
    try:
        device_info = get_device_info()
    except Exception as e:
        logger.warning(f"GPU 감지 실패, CPU 사용: {e}")
        device_info = {
            'has_gpu': False, 'gpu_count': 0, 'gpu_memory_gb': 0, 
            'gpu_name': '', 'device': torch.device('cpu')
        }
    
    cfg = optimize_config_for_data_size(cfg, n_images, device_info)
    device = device_info['device']
    
    # �️ 안전한 디바이스 설정 (GPU 실패 시 무조건 CPU)
    if cfg.fallback_to_cpu:
        try:
            # GPU 사용 가능성 재확인
            if device.type == "cuda":
                test_tensor = torch.zeros(1).to(device)
                del test_tensor
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"🛡️ GPU 테스트 실패, CPU로 안전 전환: {e}")
            device = torch.device("cpu")
            device_info['device'] = device
            device_info['has_gpu'] = False
    
    # 강제 GPU 사용은 이제 선택적
    if cfg.force_gpu and torch.cuda.is_available() and device.type == "cpu":
        logger.info("🔥 GPU 강제 사용 모드 시도...")
        try:
            device = torch.device("cuda:0")
            test_tensor = torch.zeros(1).to(device)
            del test_tensor
            torch.cuda.empty_cache()
            device_info['device'] = device
            device_info['has_gpu'] = True
            logger.info("✅ GPU 강제 사용 성공")
        except Exception as e:
            logger.warning(f"❌ GPU 강제 사용 실패, CPU 유지: {e}")
            device = torch.device("cpu")

    # 2) Metadata: prefilters + density + (optional) OCR text
    _cb("meta", 0.05, "메타데이터 수집 시작 (pHash/PDQ + density)")
    t_meta0 = time.time()
    phashes: Dict[str, imagehash.ImageHash] = {}
    pdqs: Dict[str, Optional[np.ndarray]] = {}
    densities: Dict[str, float] = {}
    texts: Dict[str, str] = {}

    images_summary_path = os.path.join(output_dir, "images_summary.csv")
    # 기존 요약 파일(images_summary.csv)이 최신이면 밀도 계산을 건너뛸 수 있도록 로드합니다
    if _is_fresh(images_summary_path, paths):
        try:
            img_df_prev = pd.read_csv(images_summary_path)
            for _, row in img_df_prev.iterrows():
                fname = row.get("파일")
                if fname in files:
                    densities[fname] = float(row.get("밀도", 0.0))
            logger.info("새 images_summary.csv 로드됨 -> 캐시 항목의 밀도 재계산 생략")
        except Exception:
            pass

    # 워커 인자 준비 및 ThreadPoolExecutor에서 실행 (IO 바운드 작업)
    worker_args = [(f, p, cfg) for f, p in zip(files, paths)]
    # 성능 최적화: 더 많은 워커로 메타데이터 수집 가속화
    # 소규모 전용: 워커 수를 과도하게 늘리지 않고 안전한 상한(4)만 사용합니다.
    max_workers = min(4, max(1, int(cfg.num_workers or 1)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for f, ph, pdqv, dens, txt in ex.map(_metadata_worker, worker_args):
            if ph is not None:
                phashes[f] = ph
            if pdqv is not None:
                pdqs[f] = pdqv
            # 새 요약에서 로드된 값이 없을 때만 density를 덮어씌움
            if f not in densities or densities.get(f, 0.0) == 0.0:
                densities[f] = dens
            if txt:
                texts[f] = txt

    blank_flags, blank_thresholds, auto_blank_threshold = _build_blank_flags(files, densities, cfg)

    # images_summary.csv를 보존(밀도 + 빈칸 플래그)하여 이후 실행을 빠르게 함
    try:
        auto_cols = []
        for f in files:
            if auto_blank_threshold is not None and os.path.splitext(f)[0].endswith(cfg.blank_auto_suffix):
                auto_cols.append(auto_blank_threshold)
            else:
                auto_cols.append(np.nan)
        img_df = pd.DataFrame({
            "파일": files,
            "밀도": [densities.get(f, 0.0) for f in files],
            "적용임계값": [blank_thresholds.get(f, cfg.blank_density_thresh) for f in files],
            "자동보정임계값": auto_cols,
            "빈칸여부": [bool(blank_flags.get(f, False)) for f in files],
        })
        img_df.to_csv(images_summary_path, index=False, encoding="utf-8-sig")
    except Exception:
        warnings.warn("Failed to write images_summary.csv")

    t_meta1 = time.time()
    _cb("meta", 0.20, f"메타데이터 완료 ({round(t_meta1 - t_meta0, 2)}s)")

    # 3) 임베딩
    # logger.info("[2/5] CNN/ViT 임베딩 처리 …")  # 간소화
    t_emb0 = time.time()
    _cb("embed", 0.22, "임베딩 계산 시작")
    
    # 캐시된 embeddings.npy 및 ordered_paths.txt 재사용 시도
    # logger.info("[2/5] CNN/ViT 임베딩 …")  # 간소화
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
                logger.warning("아티팩트 크기 불일치: 임베딩 재계산 강제")
                embs = None
                ordered_paths = None
            else:
                logger.info("캐시된 embeddings.npy 및 ordered_paths.txt 로드 완료")
        except Exception as e:
            logger.warning(f"임베딩 아티팩트 로드 실패: {e}; 재계산 예정")
            embs = None
            ordered_paths = None

    if embs is None:
        try:
            embs, ordered_paths, model_load_s = compute_embeddings(
                paths, device, cfg.batch_size, cfg.num_workers, 
                cfg.roi_ratio, cfg.embed_backend, 
                progress_callback=_cb, force_gpu=cfg.force_gpu
            )
        except Exception as e:
            logger.error(f"임베딩 계산 실패: {e}")
            if device.type == "cuda" and cfg.fallback_to_cpu:
                logger.info("🆘 CPU로 재시도...")
                device = torch.device("cpu")
                embs, ordered_paths, model_load_s = compute_embeddings(
                    paths, device, cfg.batch_size, 0,  # num_workers=0 for stability
                    cfg.roi_ratio, "resnet18",  # 안전한 백엔드
                    progress_callback=_cb, force_gpu=False
                )
            else:
                raise
        try:
            np.save(emb_art, embs)
            with open(opaths_art, "w", encoding="utf-8") as fw:
                fw.write("\n".join(ordered_paths))
        except Exception:
            warnings.warn("임베딩 아티팩트 저장 실패")
    t_emb1 = time.time()
    _cb("embed", 0.50, f"임베딩 완료 ({round(t_emb1 - t_emb0, 2)}s)")

    n = len(files)
    name_by_row = {i: os.path.basename(ordered_paths[i]) for i in range(n)}

    # 4) ANN candidates
    # logger.info("[3/5] ANN을 통한 후보 이웃 검색 …")  # 간소화
    t_ann0 = time.time()
    _cb("ann", 0.60, "ANN 후보 검색 시작")
    idxs, sims, backend_used = build_candidates(embs, cfg.k, cfg.ann_backend, cfg.hnsw_M, cfg.hnsw_efC, cfg.hnsw_efS)
    t_ann1 = time.time()
    _cb("ann", 0.78, f"ANN 완료 ({round(t_ann1 - t_ann0, 2)}s) via {backend_used}")

    # 5) Pairwise scoring → "확정 유사" 에지 만들기 → (Blossom) 최대가중치매칭으로 2장 그룹화
    # logger.info("[4/5] 쌍 점수 산정 및 페어링(최대 가중치 매칭) …")  # 간소화
    _cb("pairing", 0.80, "페어링/유사도 계산 시작")

    # 공통 페어링/그룹화 로직으로 대체
    pair_rows, groups = _pair_and_group(
        name_by_row,
        idxs,
        sims,
        phashes,
        pdqs,
        densities,
        blank_flags,
        texts,
        cfg,
        input_dir=input_dir,
    )

    # 공통 리포트 저장/파일 복사 헬퍼 호출 (타이밍 콜백 보존)
    # logger.info("[5/5] 리포트 저장 및 출력 정리 …")  # 간소화
    t_io0 = time.time()
    _cb("save", 0.95, "리포트 저장 및 파일 분류 중")
    _save_reports_and_copy(
        output_dir,
        files,
        densities,
        cfg,
        blank_flags,
        blank_thresholds,
        pair_rows,
        groups,
        input_dir=input_dir,
        embs=embs,
        backend_used=backend_used,
        auto_blank_threshold=auto_blank_threshold,
    )
    t_io1 = time.time()
    _cb("save", 0.98, f"저장 완료 ({round(t_io1 - t_io0, 2)}s)")

    _cb("finalizing", 0.995, "최종 정리 중")
    # 성능 로깅: 시간 수집 및 CSV에 추가
    try:
            # 임베딩이 캐시에서 로드된 경우 model_load_s가 설정되지 않을 수 있으므로 0으로 디폴트
        model_load_s_val = float(locals().get('model_load_s', 0.0))
        # embed_forward_s: 전방 전달에 소요된 시간(임베딩 전체 시간에서 모델 로드 시간 제외)
        total_embed_s = round(float(t_emb1 - t_emb0), 4) if 't_emb0' in locals() and 't_emb1' in locals() else 0.0
        embed_forward_s = max(0.0, total_embed_s - model_load_s_val)
        times = {
            "meta_s": round(float(t_meta1 - t_meta0), 4) if 't_meta0' in locals() and 't_meta1' in locals() else 0.0,
            # include both model load and forward pass separated
            "model_load_s": round(model_load_s_val, 4),
            "embed_forward_s": round(embed_forward_s, 4),
            "embed_s": round(total_embed_s, 4),
            "ann_s": round(float(t_ann1 - t_ann0), 4) if 't_ann0' in locals() and 't_ann1' in locals() else 0.0,
            "io_s": round(float(t_io1 - t_io0), 4) if 't_io0' in locals() and 't_io1' in locals() else 0.0,
            "total_s": round(float(time.time() - t_meta0), 4) if 't_meta0' in locals() else 0.0,
        }
        perf_row = _collect_run_features(paths, cfg, times)
        _append_perf_csv(output_dir, perf_row)
    except Exception:
        logger.warning("Failed to append perf log")

    return pair_rows, groups


def estimate_pipeline_time(input_dir_or_paths, cfg: Optional[DetectorConfig] = None,
                           recursive: bool = False, sample_size: int = 8) -> Dict:
    """
    휴리스틱과 선택적인 샘플링을 사용하여 파이프라인의 **실제 소요 시간(wall-time)**을 추정합니다.
    GPU 사양을 고려하여 더 정확한 예상치를 제공합니다.

    각 단계별 예상치를 담은 딕셔너리를 반환합니다: n_images, meta_s, embed_s, ann_s, io_s, total_s, notes
    """
    cfg = cfg or DetectorConfig()
    
    # 입력 경로 해석
    paths: List[str] = []
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    if isinstance(input_dir_or_paths, (list, tuple)):
        for p in input_dir_or_paths:
            if os.path.isfile(p) and p.lower().endswith(exts):
                paths.append(p)
    else:
        root = input_dir_or_paths
        if not os.path.exists(root):
            return {"n_images": 0, "meta_s": 0.0, "embed_s": 0.0, "ann_s": 0.0, "io_s": 0.0, "total_s": 0.0, "notes": "no input"}
        if recursive:
            for dirpath, _dirs, files in os.walk(root):
                for fn in files:
                    if fn.lower().endswith(exts):
                        paths.append(os.path.join(dirpath, fn))
        else:
            for fn in sorted(os.listdir(root)):
                if fn.lower().endswith(exts):
                    paths.append(os.path.join(root, fn))

    N = len(paths)
    if N == 0:
        return {"n_images": 0, "meta_s": 0.0, "embed_s": 0.0, "ann_s": 0.0, "io_s": 0.0, "total_s": 0.0, "notes": "no images"}

    # GPU 정보 기반 휴리스틱 개선
    device_info = get_device_info()
    has_gpu = device_info['has_gpu']
    gpu_memory_gb = device_info['gpu_memory_gb']
    gpu_name = device_info['gpu_name']
    
    # 메타데이터 처리 시간 (이미지당)
    meta_per = 0.02
    if getattr(cfg, 'use_ocr', False):
        meta_per += 0.25

    # 임베딩 처리 시간 (GPU 사양별로 세분화)
    backend = getattr(cfg, 'embed_backend', 'dinov2')
    if has_gpu:
        if 'RTX' in gpu_name or 'GTX 1660' in gpu_name:  # GTX 1660 Ti 포함
            # 중급 GPU
            emb_per = 0.015 if backend == 'dinov2' else 0.003
        elif 'GTX' in gpu_name or 'RTX 20' in gpu_name:
            # 저급~중급 GPU
            emb_per = 0.025 if backend == 'dinov2' else 0.005
        else:
            # 기타 GPU
            emb_per = 0.02 if backend == 'dinov2' else 0.005
    else:
        # CPU only
        emb_per = 0.12 if backend == 'dinov2' else 0.02

    k = getattr(cfg, 'k', 20)
    ann_s = max(0.2, 0.00012 * N * max(1, k))
    io_per = 0.008
    notes = f'heuristic+{gpu_name if has_gpu else "CPU"}'

    # 소규모 대상에서는 샘플링을 통한 실제 측정 대신 히ュー리스틱 값을 사용합니다.
    # (샘플링은 compute_embeddings를 호출하여 무거운 연산을 수행하므로 제거)

    meta_s = meta_per * N
    embed_s = emb_per * N
    io_s = io_per * N
    total_s = meta_s + embed_s + ann_s + io_s

    result = {"n_images": N, "meta_s": float(meta_s), "embed_s": float(embed_s), "ann_s": float(ann_s), "io_s": float(io_s), "total_s": float(total_s), "notes": notes}

    # 소규모 전용: 학습된 성능 모델 보정 로직 제거(복잡도 및 외부 의존성 제거)

    return result


def detect_pipeline_files(file_paths: List[str], output_dir: str,
                          config: Optional[DetectorConfig] = None,
                          progress_callback: Optional[callable] = None):
    """
    detect_pipeline 함수와 유사하지만, 이미지 파일 경로 목록을 직접 받습니다.
    file_paths: 이미지 파일의 절대 또는 상대 경로 목록.
    """
    cfg = config or DetectorConfig()

    def _cb(stage: str, pct: float = 0.0, msg: str = ""):
        # 안전하게 외부 콜백을 래핑
        try:
            if progress_callback is not None:
                progress_callback(stage=stage, pct=float(pct), msg=str(msg))
        except Exception:
            try:
                logger.debug(f"프로세스 콜백 실패: {stage} {pct} {msg}")
            except Exception:
                pass

    # Normalize and filter existing files
    paths = [os.path.abspath(p) for p in file_paths if os.path.isfile(p)]
    if not paths:
        raise FileNotFoundError("유효한 이미지 파일이 제공되지 않았습니다.")

    # 데이터 크기 및 GPU에 따른 설정 최적화
    n_images = len(paths)
    device_info = get_device_info()
    cfg = optimize_config_for_data_size(cfg, n_images, device_info)
    device = device_info['device']
    
    # 🚀 GPU 강제 사용 추가 체크
    if cfg.force_gpu and torch.cuda.is_available() and device.type == "cpu":
        logger.warning("🔥 GPU 강제 사용 모드: CPU에서 GPU로 전환합니다!")
        device = torch.device("cuda:0")
        device_info['device'] = device
        device_info['has_gpu'] = True
        # GPU 메모리 정리
        torch.cuda.empty_cache()

    # Prepare output dirs (same behavior as detect_pipeline)
    if not _safe_recreate_dir(output_dir, retries=3, delay=0.2):
        raise RuntimeError(f"출력 디렉토리 준비 실패: {output_dir}")
    for sub in ["grouped", "ok", "blank_answers", "artifacts"]:
        if not _safe_recreate_dir(os.path.join(output_dir, sub), retries=2, delay=0.1):
            logger.warning(f"하위 디렉토리 생성 실패에도 불구하고 진행합니다: {sub}")
    if not _safe_recreate_dir(os.path.join(output_dir, "artifacts", "thumbnails"), retries=2, delay=0.1):
        logger.warning("썸네일 디렉토리 생성 실패; 계속 진행합니다")

    # 파일: 파일 이름(basenames) 리스트 (보고서에 사용됨)와 '파일 이름 → 전체 경로' 매핑 정보.
    files = [os.path.basename(p) for p in paths]
    path_map = {os.path.basename(p): p for p in paths}

    # 파이프라인의 나머지 부분은 **'files'**와 **'paths'라는 이름의 리스트를 기대하며, 
    # $\text{paths}$는 $\text{files}$의 항목과 일치하는 전체 경로입니다.
    # 변수 이름을 재사용하여 $\text{detect_pipeline}$의 로직을 상당 부분 재활용합니다.

    # 2) 데이터 처리 파이프라인에서 관리되는 메타데이터의 구성 요소를 설명
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
            print("새 images_summary.csv 로드됨 -> 캐시 항목의 밀도 재계산 생략")
        except Exception:
            pass

    worker_args = [(f, path_map[f], cfg) for f in files]
    # 소규모 전용: 메타데이터 워커 수를 최대 4로 제한합니다.
    max_workers = min(4, max(1, int(cfg.num_workers or 1)))
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

    blank_flags, blank_thresholds, auto_blank_threshold = _build_blank_flags(files, densities, cfg)

    # 3) 임베딩
    logger.info("[2/5] CNN/ViT 임베딩 처리 …")
    # 이미 device 정보는 위에서 설정됨
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
                logger.warning("아티팩트 크기 불일치: 임베딩 재계산 강제")
                embs = None
                ordered_paths = None
            else:
                logger.info("캐시된 embeddings.npy + ordered_paths.txt 로드됨")
        except Exception as e:
            logger.warning(f"임베딩 아티팩트 로드 실패: {e}; 재계산 수행")
            embs = None

            ordered_paths = None

    if embs is None:
        embs, ordered_paths, model_load_s = compute_embeddings(
            [path_map[f] for f in files], device, cfg.batch_size, cfg.num_workers, 
            cfg.roi_ratio, cfg.embed_backend, 
            progress_callback=_cb, force_gpu=cfg.force_gpu
        )
        try:
            np.save(emb_art, embs)
            with open(opaths_art, "w", encoding="utf-8") as fw:
                fw.write("\n".join(ordered_paths))
        except Exception:
            warnings.warn("임베딩 아티팩트 저장 실패")

    n = len(files)
    name_by_row = {i: os.path.basename(ordered_paths[i]) for i in range(n)}

    # 4) ANN candidates
    logger.info("[3/5] ANN을 통한 후보 이웃 검색 …")
    idxs, sims, backend_used = build_candidates(embs, cfg.k, cfg.ann_backend, cfg.hnsw_M, cfg.hnsw_efC, cfg.hnsw_efS)

    # 5) Pairwise scoring → reuse same grouping logic but using path_map when needed
    logger.info("[4/5] 쌍 점수 산정 및 페어링(최대 가중치 매칭) …")

    # 공통 페어링/그룹화 로직으로 대체
    pair_rows, groups = _pair_and_group(
        name_by_row,
        idxs,
        sims,
        phashes,
        pdqs,
        densities,
        blank_flags,
        texts,
        cfg,
        path_map=path_map,
    )

    # 공통 리포트 저장/파일 복사 헬퍼 호출
    _save_reports_and_copy(
        output_dir,
        files,
        densities,
        cfg,
        blank_flags,
        blank_thresholds,
        pair_rows,
        groups,
        path_map=path_map,
        embs=embs,
        backend_used=backend_used,
        auto_blank_threshold=auto_blank_threshold,
    )

    return pair_rows, groups