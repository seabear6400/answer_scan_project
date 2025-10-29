import os
import sys
import hashlib
import re
import importlib
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import base64
import argparse

import streamlit as st
import polars as pl
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import cv2
import time
import logging
import shutil
import stat

# 모듈 로거
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Pillow resample 상수 호환
try:
    RESAMPLE = Image.Resampling.LANCZOS
except Exception:
    RESAMPLE = Image.LANCZOS


def parse_streamlit_args():
    """Streamlit 실행 시 `--` 이후의 사용자 인자를 파싱합니다.
    (예: streamlit run app.py -- --output_dir=...)
    """
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        user_args = sys.argv[idx + 1:]
    else:
        user_args = []
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--base_dir', default=None)
    p.add_argument('--default_result', default=None)
    try:
        ns, _ = p.parse_known_args(user_args)
    except SystemExit:
        class X:
            output_dir = None
            base_dir = None
            default_result = None

        ns = X()
    return ns


def _request_rerun() -> None:
    """Streamlit에서 페이지를 다시 실행하도록 요청합니다. 환경에 따라 적절한 API를 선택합니다."""
    if hasattr(st, "rerun") and callable(st.rerun):
        try:
            st.rerun()
        except Exception:
            pass
    elif hasattr(st, "experimental_rerun") and callable(st.experimental_rerun):
        try:
            st.experimental_rerun()
        except Exception:
            pass
    else:
        stop_fn = getattr(st, "stop", None)
        if callable(stop_fn):
            try:
                stop_fn()
            except Exception:
                pass


def _normalize_base_dir(raw, selection_root: Optional[Path]) -> Path:
    if raw is None:
        if selection_root is not None:
            return selection_root
        return Path.cwd()
    candidate = Path(raw)
    try:
        candidate = candidate.expanduser().resolve()
    except Exception:
        candidate = candidate.expanduser()
    if selection_root is not None:
        try:
            candidate.relative_to(selection_root)
            return selection_root
        except Exception:
            pass
    while candidate.name.endswith("_결과") and candidate.parent != candidate:
        candidate = candidate.parent
        if selection_root is not None:
            try:
                candidate.relative_to(selection_root)
                return selection_root
            except Exception:
                pass
    return candidate

ns = parse_streamlit_args()
ENV_OUTPUT_DIR = os.environ.get("ANSWER_SCAN_OUTPUT_DIR")
ENV_BASE_DIR = os.environ.get("ANSWER_SCAN_BASE_DIR")
ENV_DEFAULT_RESULT = os.environ.get("ANSWER_SCAN_DEFAULT_RESULT")
ENV_SELECTION_ROOT = os.environ.get("ANSWER_SCAN_SELECTION_ROOT")

SELECTION_ROOT: Optional[Path] = None
if ENV_SELECTION_ROOT:
    try:
        SELECTION_ROOT = Path(ENV_SELECTION_ROOT).expanduser().resolve()
    except Exception:
        SELECTION_ROOT = Path(ENV_SELECTION_ROOT).expanduser()

output_arg = ns.output_dir or ENV_OUTPUT_DIR
default_arg = ns.default_result or ENV_DEFAULT_RESULT
base_arg = ns.base_dir or ENV_BASE_DIR

CLI_OUTPUT_DIR = Path(output_arg).expanduser().resolve() if output_arg else Path.cwd()
CLI_DEFAULT_RESULT = Path(default_arg).expanduser().resolve() if default_arg else None
if base_arg:
    base_candidate = Path(base_arg)
elif SELECTION_ROOT is not None:
    base_candidate = SELECTION_ROOT
elif CLI_DEFAULT_RESULT and CLI_DEFAULT_RESULT.exists():
    base_candidate = CLI_DEFAULT_RESULT.parent
else:
    base_candidate = CLI_OUTPUT_DIR

CLI_BASE_DIR = _normalize_base_dir(base_candidate, SELECTION_ROOT)

if "_cli_base_marker" not in st.session_state or st.session_state.get("_cli_base_marker") != str(CLI_BASE_DIR):
    st.session_state["result_base_dir"] = str(CLI_BASE_DIR)
    st.session_state["_cli_base_marker"] = str(CLI_BASE_DIR)
elif "result_base_dir" not in st.session_state:
    st.session_state["result_base_dir"] = str(CLI_BASE_DIR)

_base_session = Path(st.session_state["result_base_dir"]).expanduser()
try:
    _base_session = _base_session.resolve()
except Exception:
    pass
BASE_OUTPUT_DIR = _normalize_base_dir(_base_session, SELECTION_ROOT)
if st.session_state.get("result_base_dir") != str(BASE_OUTPUT_DIR):
    st.session_state["result_base_dir"] = str(BASE_OUTPUT_DIR)

if CLI_DEFAULT_RESULT and CLI_DEFAULT_RESULT.exists():
    if not CLI_DEFAULT_RESULT.is_dir():
        CLI_DEFAULT_RESULT = CLI_DEFAULT_RESULT.parent
    try:
        CLI_DEFAULT_RESULT.relative_to(BASE_OUTPUT_DIR)
    except ValueError:
        fallback_base = _normalize_base_dir(CLI_DEFAULT_RESULT.parent, SELECTION_ROOT)
        BASE_OUTPUT_DIR = fallback_base
        st.session_state["result_base_dir"] = str(BASE_OUTPUT_DIR)
        CLI_BASE_DIR = BASE_OUTPUT_DIR
        st.session_state["_cli_base_marker"] = str(CLI_BASE_DIR)


def _has_result_files(path: Path) -> bool:
    if not path.exists():
        return False
    for fname in ("report.parquet", "report.csv"):
        if (path / fname).exists():
            return True
    return False


def _looks_like_result_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if _has_result_files(path):
        return True
    if path.name.endswith("_결과"):
        marker_files = {"images_summary.csv", "report.json", "summary.csv"}
        for fname in marker_files:
            if (path / fname).exists():
                return True
        marker_dirs = {"artifacts"}
        try:
            for child in path.iterdir():
                if child.name in marker_dirs:
                    return True
        except Exception:
            pass
    return False


def _discover_result_dirs(base_dir: Path, max_depth: int = 6) -> List[Path]:
    candidates: List[Path] = []
    base_dir = base_dir.resolve()

    if _looks_like_result_dir(base_dir):
        candidates.append(base_dir)

    skip_names = {"artifacts", "thumbnails", "disp_cache"}

    for current_root, dirnames, filenames in os.walk(base_dir):
        cur_path = Path(current_root)
        try:
            depth = len(cur_path.relative_to(base_dir).parts)
        except ValueError:
            continue

        if depth > max_depth:
            dirnames[:] = []
            continue

        if cur_path != base_dir and _looks_like_result_dir(cur_path):
            candidates.append(cur_path)
            dirnames[:] = []
            continue

        dirnames[:] = [d for d in dirnames if d not in skip_names]

    unique_candidates = []
    seen = set()
    for cand in sorted(candidates):
        if str(cand) not in seen:
            unique_candidates.append(cand)
            seen.add(str(cand))

    return unique_candidates


RESULT_DIRS = _discover_result_dirs(BASE_OUTPUT_DIR)


def _result_dir_has_rescan(path: Path) -> bool:
    targets = {"유사 후보", "중복/그룹"}
    parquet = path / "report.parquet"
    if parquet.exists():
        try:
            lf = pl.scan_parquet(str(parquet)).filter(pl.col("상태").is_in(list(targets))).limit(1)
            if lf.collect(streaming=True).height > 0:
                return True
        except Exception:
            pass
    csv_path = path / "report.csv"
    if csv_path.exists():
        try:
            for chunk in pd.read_csv(csv_path, usecols=["상태"], chunksize=2000):
                
                if chunk["상태"].isin(targets).any():
                    return True
        except Exception:
            pass
    # older pipelines used a 'grouped' subfolder; ignore filesystem scan for it
    return False


def _build_result_meta(paths: List[Path]) -> Dict[str, Dict[str, bool]]:
    meta: Dict[str, Dict[str, bool]] = {}
    for path in paths:
        raw = str(path)
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = str(path)
        has_report = _has_result_files(path)
        needs_rescan = _result_dir_has_rescan(path) if has_report else False
        entry = {
            "has_report": has_report,
            "needs_rescan": needs_rescan,
        }
        meta[raw] = entry
        if resolved != raw:
            meta[resolved] = entry
    return meta


RESULT_META = _build_result_meta(RESULT_DIRS)

# ===== 페이지 설정 =====
st.set_page_config(page_title="답안지 검수 대시보드", layout="wide")
st.title("📋 답안지 스캔 검수 대시보드 (Handwriting-Optimized)")

if "selected_result_dir" in st.session_state and st.session_state["selected_result_dir"] not in [str(p) for p in RESULT_DIRS]:
    st.session_state.pop("selected_result_dir", None)

result_options = [str(p) for p in RESULT_DIRS]

if CLI_DEFAULT_RESULT:
    default_str = str(CLI_DEFAULT_RESULT)
    if default_str in result_options and "selected_result_dir" not in st.session_state:
        st.session_state["selected_result_dir"] = default_str

if "selected_result_dir" not in st.session_state and result_options:
    preferred = next((opt for opt in result_options if RESULT_META.get(opt, {}).get("has_report")), None)
    st.session_state["selected_result_dir"] = preferred or result_options[0]

def _format_result_option(path_str: str) -> str:
    p = Path(path_str)
    try:
        rel = p.relative_to(BASE_OUTPUT_DIR)
        label = str(rel) if rel.parts else str(p)
    except ValueError:
        label = str(p)
    meta = RESULT_META.get(path_str, {})
    prefix = ""
    if meta.get("needs_rescan"):
        prefix = "[재스캔] "
    elif not meta.get("has_report"):
        prefix = "[결과 대기] "
    return f"{prefix}{label}" if prefix else label

if not result_options:
    st.sidebar.warning("결과 폴더를 찾지 못했습니다. 좌측 입력에서 분석 루트를 지정한 뒤 다시 시도하세요.")
    st.stop()

if "selected_result_dir" not in st.session_state:
    st.session_state["selected_result_dir"] = result_options[0]

selected_dir_str = st.session_state.get("selected_result_dir", result_options[0])
prev_selected_dir = st.session_state.get("_last_selected_dir")
dir_changed = prev_selected_dir is not None and prev_selected_dir != selected_dir_str
st.session_state["_last_selected_dir"] = selected_dir_str

OUTPUT_DIR = Path(selected_dir_str).resolve()
REPORT_PARQUET = os.path.join(str(OUTPUT_DIR), "report.parquet")
REPORT_CSV = os.path.join(str(OUTPUT_DIR), "report.csv")
IMG_SUMMARY = os.path.join(str(OUTPUT_DIR), "images_summary.csv")
THUMB_DIR = os.path.join(str(OUTPUT_DIR), "artifacts", "thumbnails")

REPORT_BASE_COLUMNS = ["그룹ID", "상태", "파일1", "파일2", "유사도"]

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def _is_back_page(filename: str) -> bool:
    try:
        stem = os.path.splitext(os.path.basename(str(filename)))[0]
    except Exception:
        return False
    return stem.endswith("2")

if OUTPUT_DIR.exists():
    os.makedirs(os.path.join(str(OUTPUT_DIR), "artifacts"), exist_ok=True)
    os.makedirs(THUMB_DIR, exist_ok=True)
    if not _has_result_files(OUTPUT_DIR):
        st.warning("선택한 폴더에 report.csv / report.parquet 파일이 없습니다. 결과를 생성한 뒤 다시 확인하세요.")
else:
    st.warning("선택한 폴더가 존재하지 않습니다. 올바른 경로를 입력하세요.")

# --- 메인 탭 상태 관리 ---
if "main_tab" not in st.session_state:
    st.session_state["main_tab"] = "재스캔 필요"

def switch_main_tab(tab_name: str):
    """사이드바 필터 변경 시 해당 탭으로 이동"""
    st.session_state["main_tab"] = tab_name

# --- Gallery state (전체 보기 탭 전용) ---
if "gallery_limit" not in st.session_state:
    st.session_state.gallery_limit = 120  # 한 번에 보여줄 개수 초기값 (증가)
if "gallery_selected" not in st.session_state:
    st.session_state.gallery_selected = []  # 비교 선택(최대 2장)

# 전체 보기 탭 공유 상태(사이드바 → 전역 적용)
if "gallery_search" not in st.session_state:
    st.session_state.gallery_search = ""
if "gallery_sort" not in st.session_state:
    st.session_state.gallery_sort = "파일명"
if "gallery_render_mode" not in st.session_state:
    st.session_state.gallery_render_mode = "리샘플(권장)"
if "group_filter" not in st.session_state:
    st.session_state.group_filter = "전체"
if "group_view_mode" not in st.session_state:
    st.session_state.group_view_mode = "그리드(다중 썸네일)"
if "ok_view_mode" not in st.session_state:
    st.session_state.ok_view_mode = "모두 보기"

# 재스캔 탭 삭제 워크플로 상태
if "rescan_delete_mode" not in st.session_state:
    st.session_state.rescan_delete_mode = False
if "rescan_delete_targets" not in st.session_state:
    st.session_state.rescan_delete_targets = []
if "rescan_show_confirm" not in st.session_state:
    st.session_state.rescan_show_confirm = False
if "rescan_delete_feedback" not in st.session_state:
    st.session_state.rescan_delete_feedback = None


# ===== 공용 헬퍼 =====
def _file_mtime(path: str) -> float:
    try: return os.path.getmtime(path)
    except Exception: return 0.0

def _safe_image_open(path: str) -> Image.Image:
    """이미지를 안전하게 열어 PIL.Image를 반환합니다.
    - 우선 PIL.Image.open을 시도합니다.
    - 실패하면 cv2.imread 후 PIL 변환을 시도합니다.
    - 모두 실패하면 UI가 멈추지 않도록 플레이스홀더 이미지를 반환합니다.
    """
    try:
        img = Image.open(path)
        img.load()
    # 하위 처리에서 일관되게 RGB 모드로 정규화
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.debug(f"_safe_image_open: PIL 열기 실패 {path}: {e}")
    # PIL 실패, cv2 대체 방법 시도
    try:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is not None:
            # BGR(A) -> RGB(A) 변환
            if len(arr.shape) == 3:
                if arr.shape[2] == 4:
                    b, g, r, a = cv2.split(arr)
                    arr = cv2.merge((r, g, b, a))
                    pil = Image.fromarray(arr)
                else:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(arr)
            else:
                pil = Image.fromarray(arr)
            return pil
    except Exception as e:
        logger.debug(f"_safe_image_open: cv2 대체 실패 {path}: {e}")

    # 최종 대체: UI가 멈추지 않도록 플레이스홀더 이미지 생성
    try:
        w, h = 640, 480
        ph = Image.new("RGB", (w, h), (220, 220, 220))
        draw = ImageDraw.Draw(ph)
        basename = os.path.basename(path) if path else "unknown"
        txt = f"읽을 수 없음\n{basename}"
        draw.text((8, 8), txt, fill=(80, 80, 80))
        return ph
    except Exception:
        # 플레이스홀더 생성도 실패하면 원래 예외를 재발생시킴
        raise

# ===== 캐싱: 데이터 읽기 =====
@st.cache_data(show_spinner=False)
def load_report(
    report_parquet: str,
    report_csv: str,
    columns: Optional[List[str]] = None,
    cache_token: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    _ = cache_token
    if os.path.exists(report_parquet):
        lf = pl.scan_parquet(report_parquet)
        if columns:
            available_cols = [col for col in lf.columns if col in columns]
            if available_cols:
                lf = lf.select(available_cols)
        df_out = lf.collect(streaming=True).to_pandas(use_pyarrow_extension_array=True)
        if columns:
            ordered = [col for col in columns if col in df_out.columns]
            if ordered:
                df_out = df_out[ordered]
        return df_out
    if os.path.exists(report_csv):
        try:
            if columns:
                df_csv = pd.read_csv(report_csv, usecols=lambda c: c in set(columns))
                ordered = [col for col in columns if col in df_csv.columns]
                if ordered:
                    df_csv = df_csv[ordered]
                return df_csv
            return pd.read_csv(report_csv)
        except Exception as e:
            # 빈 CSV 파일 등으로 인해 pandas가 실패하는 경우를 안전하게 처리합니다.
            # 빈 파일이면 컬럼 정보가 없으므로 빈 DataFrame을 반환해 UI가 멈추지 않도록 합니다.
            try:
                # pandas EmptyDataError를 포함한 모든 읽기 실패는 빈 DataFrame으로 처리
                from pandas.errors import EmptyDataError
                if isinstance(e, EmptyDataError):
                    return pd.DataFrame(columns=columns or [])
            except Exception:
                pass
            # 그 외 예외는 디버그 로깅 후 빈 DataFrame 반환
            logger.debug(f"report.csv 로드 실패 ({report_csv}): {e}")
            return pd.DataFrame(columns=columns or [])
    raise FileNotFoundError("결과 리포트 파일을 찾을 수 없습니다.")

@st.cache_data(show_spinner=False)
def load_img_summary(img_summary_csv: str, cache_buster: Optional[float] = None) -> pd.DataFrame:
    _ = cache_buster
    if os.path.exists(img_summary_csv):
        return pd.read_csv(img_summary_csv)
    return pd.DataFrame(columns=["파일", "밀도", "빈칸여부"])

def _iter_images(root: str):
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(IMAGE_EXTS):
                        yield entry.path
        except PermissionError:
            continue

@st.cache_data(show_spinner=False)
def list_all_images(root: str, cache_buster: float = 0) -> List[str]:
    """
    루트 폴더 아래의 이미지 파일을 재귀적으로 나열합니다.
    cache_buster는 외부에서 캐시를 무효화할 때 사용합니다.
    """
    try:
        # 주의: 전역 IMG_SUMMARY는 현재 선택된 OUTPUT_DIR을 가리키지만,
        # 함수 호출자가 전달한 root를 사용해 이미지 요약 파일 경로를 계산하면
        # 캐시 키가 root에 따라 분리되어 폴더 전환 시 더 안전합니다.
        summary_csv = os.path.join(root, "images_summary.csv")
        summary_df = load_img_summary(summary_csv, cache_buster=cache_buster)
        ordered_txt = os.path.join(root, "artifacts", "ordered_paths.txt")
        ordered_map: Dict[str, List[str]] = {}
        if os.path.isfile(ordered_txt):
            try:
                with open(ordered_txt, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        p = line.strip().strip('"')
                        if not p:
                            continue
                        path = os.path.normpath(p)
                        if os.path.isfile(path):
                            bn = os.path.basename(path).lower()
                            ordered_map.setdefault(bn, []).append(path)
            except Exception as exc:
                logger.debug(f"ordered_paths 로드 실패: {exc}")

        seen: Dict[str, None] = {}
        if not summary_df.empty and '파일' in summary_df.columns:
            for entry in summary_df['파일']:
                if not isinstance(entry, str) or not entry:
                    continue
                candidates: List[str] = []
                if os.path.isabs(entry):
                    candidates.append(entry)
                else:
                    candidates.append(os.path.join(root, entry))
                bn = os.path.basename(entry).lower()
                candidates.extend(ordered_map.get(bn, []))
                for cand in candidates:
                    path = os.path.normpath(cand)
                    if os.path.isfile(path):
                        seen[path] = None

        if not seen and ordered_map:
            for paths in ordered_map.values():
                for p in paths:
                    if os.path.isfile(p):
                        seen[p] = None

        if seen:
            return sorted(seen.keys())
    except Exception as exc:
        logger.debug(f"images_summary 기반 이미지 목록 활용 실패: {exc}")
    return sorted(_iter_images(root))

# ===== 캐싱: 베이스네임 → 경로 맵 (탐색/해결용) =====
@st.cache_data(show_spinner=False)
def build_basename_map(root: str, cache_buster: float = 0) -> Dict[str, str]:
    """Build basename->path map. If the same filename exists in multiple locations,
    the first discovered path is used. Priority is no longer special-cased for
    grouped/ok/blank_answers because the pipeline no longer creates those folders.
    """
    # cache_buster는 파일 변경 시 호출자가 강제 재계산하도록 허용합니다
    _ = cache_buster
    imgs = list_all_images(root, cache_buster=cache_buster)
    def pri(p: str) -> int:
        # Keep a deterministic pseudo-priority based on path depth and lexicographic order
        low = p.replace("\\", "/").lower()
        # shorter paths (closer to root) get higher priority
        try:
            parts = low.count("/")
        except Exception:
            parts = 0
        return parts
    best: Dict[str, str] = {}
    best_pri: Dict[str, int] = {}
    for p in imgs:
        bn = os.path.basename(p).lower()
        rank = pri(p)
        if (bn not in best) or (rank < best_pri[bn]):
            best[bn] = p
            best_pri[bn] = rank
    return best

SEARCH_CACHE: Dict[str, Optional[str]] = {}


def clear_caches_and_state(session_keys: Optional[List[str]] = None) -> None:
    """안전한 캐시 및 세션 상태 초기화 유틸리티

    - 최신 Streamlit API(`st.cache_data.clear`)를 우선 사용하고, 실패하면
      `st.experimental_memo.clear`로 폴백합니다.
    - 개별 데코레이터 기반 캐시 함수(list_all_images 등)에 대해 `.clear()`가
      가능하면 시도합니다.
    - session_keys에 명시된 세션 키들(예: ['gallery_selected'])을 제거합니다.
    모든 예외는 흘려보내어 대시보드가 중단되지 않도록 설계했습니다.
    """
    try:
        # 최신 Streamlit 전역 캐시 우선 삭제
        if hasattr(st, "cache_data") and callable(getattr(st.cache_data, "clear", None)):
            try:
                st.cache_data.clear()
            except Exception:
                # 내부 구현 차이로 실패할 수 있음; 폴백으로 계속 진행
                pass
    except Exception:
        pass

    try:
        # 구버전 Streamlit 호환
        if hasattr(st, "experimental_memo") and callable(getattr(st.experimental_memo, "clear", None)):
            try:
                st.experimental_memo.clear()
            except Exception:
                pass
    except Exception:
        pass

    # 개별 함수별 clear() 시도 (존재하면 안전하게 호출)
    for _fn in (globals().get('load_img_summary'), globals().get('list_all_images'), globals().get('build_basename_map')):
        try:
            if _fn and hasattr(_fn, 'clear') and callable(getattr(_fn, 'clear')):
                try:
                    _fn.clear()
                except Exception:
                    pass
        except Exception:
            pass

    # SEARCH_CACHE 등 모듈 레벨 캐시 초기화
    try:
        SEARCH_CACHE.clear()
    except Exception:
        pass

    # 세션 상태 키 제거(옵션)
    if session_keys:
        for k in session_keys:
            try:
                if k in st.session_state:
                    del st.session_state[k]
            except Exception:
                pass

    # 일반적인 UI 관련 키 제거(안전하게 시도)
    for k in ("gallery_limit", "gallery_selected"):
        try:
            if k in st.session_state:
                del st.session_state[k]
        except Exception:
            pass

if dir_changed:
    # 선택한 결과 폴더가 바뀐 경우, 안전한 유틸리티를 통해 캐시와 관련 세션 상태를 초기화합니다.
    # 상세 무효화 로직은 `clear_caches_and_state`에 위임됩니다.
    try:
        clear_caches_and_state(session_keys=["gallery_selected", "gallery_limit"])
    except Exception:
        # 극단적 예외가 발생하면 최소한 기본 상태만 초기화
        try:
            st.session_state.gallery_limit = 120
        except Exception:
            pass
        try:
            st.session_state.gallery_selected = []
        except Exception:
            pass
        try:
            SEARCH_CACHE.clear()
        except Exception:
            pass

BASENAME_MAP = build_basename_map(str(OUTPUT_DIR))


def _candidate_roots() -> List[str]:
    """이미지를 탐색할 후보 루트 디렉터리를 우선순위 순으로 반환합니다."""
    raw: List[Optional[Path]] = [
        OUTPUT_DIR,
        BASE_OUTPUT_DIR,
        CLI_BASE_DIR,
        CLI_OUTPUT_DIR,
    ]
    if SELECTION_ROOT is not None:
        raw.append(SELECTION_ROOT)
    try:
        parent = OUTPUT_DIR.parent
        if parent != OUTPUT_DIR:
            raw.append(parent)
    except Exception:
        pass

    roots: List[str] = []
    seen: set[str] = set()
    for cand in raw:
        if cand is None:
            continue
        try:
            cand_str = str(cand)
        except Exception:
            continue
        try:
            abs_path = os.path.abspath(cand_str)
        except Exception:
            continue
        if abs_path in seen:
            continue
        if os.path.isdir(abs_path):
            seen.add(abs_path)
            roots.append(abs_path)
    return roots


def _normalize_lookup_key(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    normalized = re.sub(r"/+", "/", normalized)
    return f"{OUTPUT_DIR}|{normalized.lower()}"


def _search_basename_in_root(root: str, bn_lower: str, max_depth: int = 6) -> Optional[str]:
    if not bn_lower:
        return None
    try:
        root_path = Path(root)
    except Exception:
        return None
    if not root_path.exists():
        return None

    skip_dirs = {"disp_cache", "__pycache__", ".git", ".venv"}
    try:
        for current_root, dirnames, filenames in os.walk(root):
            try:
                depth = len(Path(current_root).relative_to(root_path).parts)
            except Exception:
                depth = 0
            if depth > max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fname in filenames:
                if fname.lower() == bn_lower:
                    return os.path.normpath(os.path.join(current_root, fname))
    except Exception as exc:
        logger.debug(f"베이스네임 검색 실패: {root} / {bn_lower}: {exc}")
    return None


def resolve_image_path(name_or_path: str) -> Optional[str]:
    """
    - 절대/상대 경로가 유효하면 그대로 사용
    - 아니면 OUTPUT_DIR 하위에서 파일명으로 검색(베이스네임 맵 사용)
    """
    if not name_or_path:
        return None
    raw = name_or_path.strip()
    if not raw:
        return None

    def _cache_and_return(cache_key: str, value: Optional[str]) -> Optional[str]:
        SEARCH_CACHE[cache_key] = value
        return value

    candidate_roots: Optional[List[str]] = None

    def _roots() -> List[str]:
        nonlocal candidate_roots
        if candidate_roots is None:
            candidate_roots = _candidate_roots()
        return candidate_roots

    # 직접 경로 우선 검사
    try:
        expanded = os.path.expanduser(raw)
    except Exception:
        expanded = raw
    if os.path.isfile(expanded):
        normed = os.path.normpath(expanded)
        return _cache_and_return(_normalize_lookup_key(raw), normed)

    rel = os.path.normpath(os.path.join(str(OUTPUT_DIR), raw))
    if os.path.isfile(rel):
        return _cache_and_return(_normalize_lookup_key(raw), rel)

    cache_key = _normalize_lookup_key(raw)
    if cache_key in SEARCH_CACHE:
        return SEARCH_CACHE[cache_key]

    bn = os.path.basename(raw).lower()

    if bn:
        mapped = BASENAME_MAP.get(bn)
        if mapped and os.path.isfile(mapped):
            return _cache_and_return(cache_key, mapped)

    # 상대 경로를 각 후보 루트와 결합해 확인
    if not os.path.isabs(raw):
        sanitized = raw.lstrip("./\\")
        for root in _roots():
            joined = os.path.normpath(os.path.join(root, sanitized))
            if os.path.isfile(joined):
                return _cache_and_return(cache_key, joined)

    if bn:
        try:
            input_map = load_input_basename_map()
            for cand in input_map.get(bn, []):
                if os.path.isfile(cand):
                    return _cache_and_return(cache_key, os.path.normpath(cand))
        except Exception:
            pass

        for root in _roots():
            root_key = f"root::{root}|{bn}"
            if root_key in SEARCH_CACHE:
                cached = SEARCH_CACHE[root_key]
                if cached:
                    return _cache_and_return(cache_key, cached)
                continue
            found = _search_basename_in_root(root, bn)
            SEARCH_CACHE[root_key] = found
            if found:
                return _cache_and_return(cache_key, found)

    return _cache_and_return(cache_key, None)


# ===== 번호/앞뒷장 유틸 =====
def _extract_first_number(s: str):
    """문자열에서 첫 번째 연속 숫자 그룹을 찾아 (숫자문자열, 정수값, start, end) 반환하거나 None 반환."""
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    ns = m.group(1)
    return ns, int(ns), m.start(1), m.end(1)

def corresponding_front_filename(fname: str) -> str:
    """파일명에서 숫자를 찾아 짝수이면 -1(앞면), 홀수면 그대로로 대응하는 앞면 파일명을 생성하여 반환."""
    rec = _extract_first_number(fname)
    if not rec:
        return fname
    ns, n, sidx, eidx = rec
    if n % 2 == 0:
        front_n = n - 1
    else:
        front_n = n
    front_ns = str(front_n).zfill(len(ns))
    return fname[:sidx] + front_ns + fname[eidx:]


# 안전한 2파일(뒷장) 판별기: 파일명이 충분히 길지 않거나 예상 포맷이 아닐 때 False 반환
def is_2file(filename: str) -> bool:
    try:
        if not filename or not isinstance(filename, str):
            return False
        # 확장자 포함 길이가 최소 5 이상이어야(예: 1.jpg) 뒤에서 5번째 문자가 존재
        return filename.lower().endswith('.jpg') and len(filename) >= 5 and filename[-5] == '2'
    except Exception:
        return False

# ===== 캐싱: 표시용(썸네일/대형) 이미지 생성 =====
@st.cache_data(show_spinner=False)
def _disp_key(src_path: str, size: int, fmt: str, quality: int) -> str:
    stat = f"{src_path}|{_file_mtime(src_path)}|{size}|{fmt}|{quality}"
    return hashlib.md5(stat.encode("utf-8")).hexdigest()[:16]

def make_display_image(src_path: str, size: int, fmt: str = "WEBP", quality: int = 95) -> str:
    """긴 변을 기준으로 size(px)까지 축소한 표시용 이미지를 캐시에 생성/재사용합니다.
    LANCZOS 리샘플링을 사용하며 포맷은 WEBP/JPEG/PNG를 지원합니다.
    quality는 JPEG/WEBP에 적용됩니다. 반환값은 캐시된 파일 경로입니다. 
    """
    fmt = fmt.upper()
    ext_map = {"WEBP": "webp", "JPEG": "jpg", "PNG": "png"}
    ext = ext_map.get(fmt, "webp")
    key = _disp_key(src_path, size, fmt, quality)
    cache_sub = os.path.join(THUMB_DIR, "disp_cache")
    os.makedirs(cache_sub, exist_ok=True)
    dst = os.path.join(cache_sub, f"{key}.{ext}")

    try:
        if not os.path.exists(dst) or _file_mtime(dst) < _file_mtime(src_path):
            img = _safe_image_open(src_path).convert("RGB")
            w, h = img.size
            if max(w, h) > size:
                if w >= h:
                    new_w = size
                    new_h = max(1, int(h * (size / w)))
                else:
                    new_h = size
                    new_w = max(1, int(w * (size / h)))
                img = img.resize((new_w, new_h), RESAMPLE)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if fmt == "PNG":
                img.save(dst, "PNG", optimize=True)
            elif fmt == "JPEG":
                img.save(dst, "JPEG", quality=quality, optimize=True, progressive=True)
            else:  # WEBP
                img.save(dst, "WEBP", quality=quality, method=6)
    except Exception as e:
        logger.debug(f"make_display_image 실패: {src_path} -> {dst}: {e}")
        return src_path
    return dst

# ===== KPI 계산 =====
def compute_kpis(df: pd.DataFrame, img_df: pd.DataFrame) -> Dict[str, int]:
    kpis = {"총 이미지": 0, "그룹 수": 0, "공백 수": 0, "유사 후보 쌍": 0}

    # 총 이미지
    try:
        kpis["총 이미지"] = int(len(img_df)) if hasattr(img_df, '__len__') else 0
    except Exception:
        pass

    # 그룹 수 (report의 그룹ID 컬럼, '-'은 무시)
    try:
        if isinstance(df, pd.DataFrame) and '그룹ID' in df.columns:
            kpis["그룹 수"] = int(df['그룹ID'].replace('-', pd.NA).dropna().nunique())
    except Exception:
        pass

    # 공백 수: 이미지 요약 또는 리포트에서 파생 (뒷장 파일 기준)
    try:
        if isinstance(img_df, pd.DataFrame) and '빈칸여부' in img_df.columns:
            blanks_mask = img_df['빈칸여부'].astype(bool)
            if '파일' in img_df.columns:
                file_series = img_df.loc[blanks_mask, '파일'].astype(str)
                filtered = [name for name in file_series if _is_back_page(name)]
                kpis["공백 수"] = len(filtered)
            else:
                kpis["공백 수"] = int(blanks_mask.sum())
        elif isinstance(df, pd.DataFrame) and '상태' in df.columns:
            blanks_mask = df['상태'].astype(str).str.contains('공백', na=False)
            if blanks_mask.any():
                names: List[str] = []
                for col in ("파일1", "파일2"):
                    if col in df.columns:
                        names.extend(df.loc[blanks_mask, col].astype(str).tolist())
                filtered = {name for name in names if _is_back_page(name)}
                kpis["공백 수"] = len(filtered)
            else:
                kpis["공백 수"] = 0
    except Exception:
        pass

    # 유사 후보 쌍: report(df)의 '상태' 컬럼에서 '유사 후보'로 표기된 행 수 (있을 경우)
    try:
        if isinstance(df, pd.DataFrame) and '상태' in df.columns:
            kpis["유사 후보 쌍"] = int((df['상태'].astype(str) == '유사 후보').sum())
    except Exception:
        pass

    return kpis

# ===== 데이터 로딩 =====
report_cache_token = (_file_mtime(REPORT_PARQUET), _file_mtime(REPORT_CSV))
try:
    df = load_report(
        REPORT_PARQUET,
        REPORT_CSV,
        columns=REPORT_BASE_COLUMNS,
        cache_token=report_cache_token,
    )
    report_available = True
except FileNotFoundError:
    report_available = False
    df = pd.DataFrame(columns=REPORT_BASE_COLUMNS)
    st.info("리포트 파일이 아직 생성되지 않았습니다. 파이프라인 실행 후 다시 선택하세요.")

img_df = load_img_summary(IMG_SUMMARY, cache_buster=_file_mtime(IMG_SUMMARY))

# ===== KPI 카드 =====
kpis = compute_kpis(df, img_df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 이미지", f"{kpis['총 이미지']:,}")
c2.metric("그룹 수", f"{kpis['그룹 수']:,}")
c3.metric("공백 수", f"{kpis['공백 수']:,}")


# ===== 사이드바: 꼭 필요한 옵션만 노출 =====
st.sidebar.header("주요 필터/설정")
# 사이드바: 결과 파일/아티팩트 존재 여부 요약
try:
    missing = []
    checks = [(REPORT_PARQUET, 'report.parquet'), (REPORT_CSV, 'report.csv'), (IMG_SUMMARY, 'images_summary.csv')]
    for p, name in checks:
        if not os.path.exists(p):
            missing.append(name)
    art_txt = os.path.join(OUTPUT_DIR, 'artifacts', 'ann_backend.txt')
    if not os.path.exists(art_txt):
        # 필수가 아닌 결과물/산출물이므로 경고는 하지 않음
        pass
    if missing:
        st.sidebar.warning("결과 파일 누락: " + ", ".join(missing) + ". 먼저 파이프라인을 실행하세요.")

except Exception:
    pass
# ===== 테마 선택 =====
THEMES = {
    'Light (기본)': {
        'palette': { 'bg':'#FBFDFF','sidebar_bg':'#FFFFFF','text':'#091223','sidebar_text':'#091223','secondary':'#475569','accent':'#0B66FF','card_bg':'#FBFDFF','card_border':'#e6eef8','shadow':'0 6px 18px rgba(10,20,40,0.04)'},
    },
    'Warm Sepia': {
        'palette': { 'bg':'#f4efe6','sidebar_bg':'#efe6d9','text':'#2d2a26','sidebar_text':'#2d2a26','secondary':'#6e5a4a','accent':'#b77936','card_bg':'#fbf6ee','card_border':'#e6dccf','shadow':'0 6px 18px rgba(30,20,10,0.08)'},
    },
    'Gentle Mint': {
        'palette': { 'bg':'#f3faf6','sidebar_bg':'#eaf7ef','text':'#082724','sidebar_text':'#082724','secondary':'#4b6b64','accent':'#39b89f','card_bg':'#ffffff','card_border':'#e6f0ec','shadow':'0 6px 18px rgba(5,30,25,0.06)'} ,
    }
}

if 'theme' not in st.session_state:
    st.session_state['theme'] = 'Light (기본)'
def _inject_theme_css(mode: str = 'Light (기본)'):
    # mode에 따라 팔레트 선택
    theme = THEMES.get(mode, THEMES['Light (기본)'])
    pal = theme['palette']
    sidebar_width = int(st.session_state.get('sidebar_width_px', 350))

    # 기본값 보장
    bg = pal.get('bg','#F7F9FB')
    sidebar_bg = pal.get('sidebar_bg', '#FFFFFF')
    text = pal.get('text', '#0B1726')
    sidebar_text = pal.get('sidebar_text', text)
    secondary_text = pal.get('secondary', '#41515F')
    accent = pal.get('accent', '#0B66FF')
    card_bg = pal.get('card_bg', '#FFFFFF')
    card_border = pal.get('card_border', '#e6e9ee')
    shadow = pal.get('shadow', 'none')

    css = f"""
    <style>
    .stApp {{ background-color: {bg} !important; color: {text} !important; }}
    [data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; box-shadow: none !important; color: {sidebar_text} !important; }}
    [data-testid="stSidebar"][aria-expanded="true"] {{ width: {sidebar_width}px !important; min-width: {sidebar_width}px !important; }}
    [data-testid="stSidebar"][aria-expanded="false"] {{ width: 0 !important; min-width: 0 !important; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stHeader, [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .css-1d391kg {{ color: {sidebar_text} !important; opacity: 0.98 !important; }}
    .stBlock, .stCard {{ background-color: {card_bg} !important; border: 1px solid {card_border}; border-radius: 10px; box-shadow: {shadow}; padding: 12px; }}
    .stMetric {{ color: {text} !important; }}
    /* KPI/Metric 내부 텍스트(라벨/서브텍스트)가 다크에서 안보이는 문제 해결: 강제 색상/불투명도 적용 */
    .stMetric, .stMetric * {{ color: {text} !important; opacity: 0.98 !important; }}
    .stMetric p, .stMetric span, .stMetric small {{ color: {secondary_text} !important; opacity: 0.95 !important; }}
    input, textarea, select, button {{ color: {text} !important; background-color: transparent !important; border-radius: 8px; }}
    .stApp p, .stApp span, label, .css-1v0mbdj p {{ color: {secondary_text} !important; }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{ color: {sidebar_text} !important; }}
    a, .stButton>button, .css-18e3th9 a, .css-18e3th9 button {{ color: {accent} !important; }}
    .stDataFrame table {{ border-collapse: separate; border-spacing: 0 8px; }}
    img {{ border-radius: 8px; box-shadow: 0 8px 24px rgba(2,8,12,0.15); }}
    /* 검색 입력 상자 강조: 사용자 요청으로 가독성 향상용 추가 스타일입니다. */
    input[type="text"], .stTextInput>div>div>input {{
        background-color: rgba(255,255,255,0.9) !important;
        border: 1.5px solid {accent} !important;
        box-shadow: 0 4px 10px rgba(11,102,255,0.08) !important;
        padding: 10px 12px !important;
        border-radius: 10px !important;
        font-size: 14px !important;
        color: {text} !important;
    }}
    /* 사이드바 내 입력과 플레이스홀더 대비 개선 */
    [data-testid="stSidebar"] input[type="text"] {{ 
        background-color: rgba(255,255,255,0.95) !important; 
        color: {text} !important;
    }}
    input::placeholder, textarea::placeholder {{ color: rgba(0,0,0,0.38) !important; font-weight: 500 !important; }}
    
    /* 사이드바 select 박스 - 심플하고 깔끔한 스타일 */
    [data-testid="stSidebar"] .stSelectbox>div>div {{
        background-color: rgba(255,255,255,0.98) !important;
        border: 1px solid rgba(11,102,255,0.3) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
        transition: border-color 0.2s ease !important;
    }}
    
    [data-testid="stSidebar"] .stSelectbox>div>div:hover {{
        border-color: {accent} !important;
        box-shadow: 0 2px 8px rgba(11,102,255,0.12) !important;
    }}
    
    [data-testid="stSidebar"] .stSelectbox>div>div>div {{
        color: {text} !important;
        font-weight: 500 !important;
        padding: 10px 12px !important;
        font-size: 14px !important;
    }}

    /* 닫힌 상태 표시 박스 스타일은 기본으로 유지합니다. (사용자 요청: 하단은 회색 적용 안 함) */
    
    /* select 드롭다운 화살표 스타일링 */
    [data-testid="stSidebar"] .stSelectbox svg {{
        color: {accent} !important;
        opacity: 0.7 !important;
    }}
    
    /* 드롭다운 옵션 리스트 스타일링 */
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] {{
        background-color: white !important;
        border: 1px solid rgba(11,102,255,0.2) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        margin-top: 2px !important;
    }}
    
    [data-testid="stSidebar"] .stSelectbox [role="option"] {{
        color: {text} !important;
        padding: 8px 12px !important;
        margin: 2px 4px !important;
        border-radius: 4px !important;
        font-size: 14px !important;
        transition: background-color 0.15s ease !important;
    }}
    
    [data-testid="stSidebar"] .stSelectbox [role="option"]:hover {{
        background-color: rgba(11,102,255,0.05) !important;
        color: {accent} !important;
    }}
    
    [data-testid="stSidebar"] .stSelectbox [aria-selected="true"] {{
        background-color: {accent} !important;
        color: white !important;
        font-weight: 500 !important;
    }}

    /* 재스캔 필요 옵션 강조 표현: BaseWeb aria-label을 활용해 매칭합니다. */
    [data-testid="stSidebar"] .stSelectbox [role="option"][aria-label^="[재스캔]"] {{
        color: #d62839 !important;
        font-weight: 600 !important;
        background-color: rgba(214,40,57,0.08) !important;
    }}

    [data-testid="stSidebar"] .stSelectbox [role="option"][aria-selected="true"][aria-label^="[재스캔]"] {{
        background-color: rgba(214,40,57,0.14) !important;
        color: #d62839 !important;
    }}

    [data-testid="stSidebar"] .stSelectbox [role="option"][aria-label^="[재스캔]"]::before {{
        content: "⚠ ";
        font-weight: 700;
    }}

    [data-testid="stSidebar"] .stSelectbox>div>div>div[aria-label^="[재스캔]"] {{
        color: #d62839 !important;
        font-weight: 600 !important;
    }}

    [data-testid="stSidebar"] .stSelectbox>div>div>div[aria-label^="[재스캔]"]::before {{
        content: "⚠ ";
        margin-right: 4px;
    }}

    /* 드롭다운 목록에서 '선택된 옵션'을 더 시각적으로 강조합니다. */
    /* 선택된 옵션은 파란색 대신 회색 배경으로 고정하여 '선택 중'을 표시합니다. */
    [data-testid="stSidebar"] .stSelectbox [role="option"][aria-selected="true"] {{
        /* 목록 내부에서 선택된 항목을 더 진한 회색으로 표시 */
        background-color: rgba(0,0,0,0.12) !important; /* 약간 더 진한 회색 */
        color: {sidebar_text} !important; /* 진한 텍스트 */
        font-weight: 700 !important;
        border-radius: 0 0 6px 6px !important;
        position: relative !important;
    }}

    /* 다양한 구현에서 선택 상태를 나타내는 속성에 모두 대응하여 회색 강조를 강제합니다. */
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] {{
        position: relative !important;
        overflow: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }}

    /* 선택 상태에 대한 공통 규칙(목록 내부에서만 적용) */
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][aria-selected="true"],
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][data-selected="true"],
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][aria-current="true"] {{
        background-color: rgba(0,0,0,0.12) !important;
        color: {sidebar_text} !important;
        font-weight: 700 !important;
    }}

    /* 선택된 항목은 목록에서 상단에 고정(sticky)되도록 함: 이미지2 스타일과 유사하게 보이게 함 */
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][aria-selected="true"],
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][data-selected="true"] {{
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 10 !important;
        margin-top: 0 !important;
    }}

    /* hover가 선택 스타일을 덮어쓰지 않도록 유지 */
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][aria-selected="true"]:hover {{
        background-color: rgba(0,0,0,0.12) !important;
        color: {sidebar_text} !important;
    }}

    /* ===== 포털(overlay)로 렌더되는 BaseWeb/Select의 드롭다운을 직접 타깃합니다. =====
       Streamlit은 드롭다운을 sidebar 바깥(포털)으로 렌더할 수 있어 기존 사이드바 내부 선택자로 매칭되지 않을 수 있습니다.
       아래 규칙은 포털 내부의 listbox/option에 대해 동일한 강조(회색 배경, 진한 텍스트, sticky)를 강제합니다. */
    .baseweb-portal [role="listbox"] [role="option"][aria-selected="true"],
    .baseweb-portal [role="listbox"] [role="option"][data-selected="true"],
    body > [role="listbox"] [role="option"][aria-selected="true"] {{
        background-color: rgba(0,0,0,0.12) !important;
        color: {sidebar_text} !important;
        font-weight: 700 !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 9999 !important;
    }}

    /* 포털 내 선택된 옵션이 hover에 의해 덮어쓰이지 않도록 함 */
    .baseweb-portal [role="listbox"] [role="option"][aria-selected="true"]:hover {{
        background-color: rgba(0,0,0,0.12) !important;
        color: {sidebar_text} !important;
    }}

    /* ------------------------------------------------------------------
       Streamlit이 생성하는 emotion 클래스(예: st-emotion-cache-xxxxx etx0m6x1)
       를 직접 타깃해 내부 텍스트 컨테이너에도 회색 배경/패딩을 강제합니다.
       - 사이드바 내부 렌더링과 포털(overlay) 렌더링을 모두 커버합니다.
       - 특정 동적 클래스명이 바뀔 수 있으므로, etx-prefixed 클래스도 함께 지정합니다.
    ------------------------------------------------------------------ */
    /* 사이드바 내부 listbox */
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][aria-selected="true"] .st-emotion-cache-qiev7j,
    [data-testid="stSidebar"] .stSelectbox [role="listbox"] [role="option"][aria-selected="true"] .etx0m6x1 {{
        background-color: rgba(0,0,0,0.12) !important;
        display: block !important;
        padding: 8px 12px !important;
        margin: -8px -12px !important; /* 옵션 컨테이너 패딩과 겹치지 않게 보정 */
        color: {sidebar_text} !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
    }}

    /* 포털(overlay)로 렌더된 listbox */
    .baseweb-portal [role="listbox"] [role="option"][aria-selected="true"] .st-emotion-cache-qiev7j,
    .baseweb-portal [role="listbox"] [role="option"][aria-selected="true"] .etx0m6x1,
    body > [role="listbox"] [role="option"][aria-selected="true"] .st-emotion-cache-qiev7j,
    body > [role="listbox"] [role="option"][aria-selected="true"] .etx0m6x1 {{
        background-color: rgba(0,0,0,0.12) !important;
        display: block !important;
        padding: 8px 12px !important;
        margin: -8px -12px !important;
        color: {sidebar_text} !important;
        font-weight: 700 !important;
        border-radius: 4px !important;
    }}

    /* 선택된 옵션에 체크 표시를 추가해 사용자가 어떤 항목이 선택됐는지 바로 알 수 있도록 함 */
    [data-testid="stSidebar"] .stSelectbox [role="option"][aria-selected="true"]::after {{
        content: "✔";
        position: absolute;
        right: 10px;
        top: 50%;
        transform: translateY(-50%);
        color: {sidebar_text} !important; /* 회색 배경에 어울리는 진한 색상 */
        font-weight: 700;
    }}

    /* 선택된 옵션 왼쪽에 컬러 바 추가하여 '현재 선택'을 시각적으로 강조 */
    /* 왼쪽 컬러 바는 회색 톤으로 변경하여 전체가 회색 강조로 보이도록 함 */
    [data-testid="stSidebar"] .stSelectbox [role="option"][aria-selected="true"]::before {{
        content: "";
        position: absolute;
        left: 6px;
        top: 8px;
        bottom: 8px;
        width: 4px;
        background: rgba(0,0,0,0.25) !important; /* 진한 회색 바 */
        border-radius: 2px;
    }}

    /* 옵션 텍스트가 왼쪽 컬러 바와 겹치지 않도록 패딩 보정 */
    [data-testid="stSidebar"] .stSelectbox [role="option"] {{
        padding-left: 18px !important;
    }}
    </style>
    """
    try:
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        pass

path_tab, theme_tab, rescan_tab, ok_tab, gallery_tab = st.sidebar.tabs(["분석 경로", "테마", "재스캔 필요", "정상/공백 답안", "전체 보기"])
quality_options_common = ["빠름", "균형", "선명"]

with path_tab:
    st.markdown("**분석 경로 설정**")
    # 텍스트 입력의 표시값은 현재의 result_base_dir을 사용합니다.
    # 버튼으로 기본 경로를 적용/복원하면 st.session_state['result_base_dir']이 변경되고
    # _request_rerun()로 재실행될 때 이 입력의 값이 갱신되어 보이게 됩니다.
    base_input = st.text_input(
        "검색 시작 경로",
        value=st.session_state.get("result_base_dir", str(BASE_OUTPUT_DIR)),
        key="result_base_input",
    )

    # 안전하게 세션 상태를 변경하기 위한 콜백 함수들
    def _apply_base_dir_cb():
        raw = st.session_state.get("result_base_input", "")
        try:
            new_base = Path(raw).expanduser()
            normalized_base = _normalize_base_dir(new_base, SELECTION_ROOT)
            st.session_state["result_base_dir"] = str(normalized_base)
            st.session_state.pop("selected_result_dir", None)
            st.session_state["_cli_base_marker"] = str(normalized_base)
        except Exception:
            # 실패 시 기존 동작 유지
            pass
        # 콜백 내부에서는 st.rerun()이 no-op일 수 있으므로 여기서는 명시적 재실행을 호출하지 않습니다.
        # 세션 상태를 변경하면 Streamlit이 콜백 종료 후 자동으로 스크립트를 재실행합니다.
        try:
            st.cache_data.clear()
        except Exception:
            pass

    def _reset_base_dir_cb():
        # CLI 기본값으로 복원: 마커과 result_base_dir만 갱신합니다.
        try:
            st.session_state["result_base_dir"] = str(CLI_BASE_DIR)
            st.session_state.pop("selected_result_dir", None)
            st.session_state["_cli_base_marker"] = str(CLI_BASE_DIR)
            # 텍스트 입력도 같은 값을 반영하도록 설정
            st.session_state["result_base_input"] = str(CLI_BASE_DIR)
        except Exception:
            pass
        # 콜백 내부에서 강제 rerun을 호출하지 않음: 세션 상태 변경으로 자동 재실행됩니다.
        try:
            st.cache_data.clear()
        except Exception:
            pass

    path_cols = st.columns(3)
    with path_cols[0]:
        st.button("경로 적용", key="apply_base_dir", on_click=_apply_base_dir_cb)
    with path_cols[1]:
        st.button("기본 경로 복원", key="reset_base_dir", on_click=_reset_base_dir_cb)
    # (목록 새로고침 버튼 제거됨)

    st.selectbox(
        "분석 결과 폴더",
        options=result_options,
        format_func=_format_result_option,
        key="selected_result_dir",
    )

with theme_tab:
    st.markdown("**대시보드 테마**")
    sidebar_width_default = int(st.session_state.get("sidebar_width_px", 350))
    sidebar_slider_args = {
        "label": "사이드바 폭",
        "min_value": 260,
        "max_value": 520,
        "key": "sidebar_width_px",
        "help": "사이드바 영역의 폭을 조정해 긴 라벨이나 컨트롤이 잘려 보이지 않도록 합니다."
    }
    st.slider(value=sidebar_width_default, **sidebar_slider_args)
    theme_keys = list(THEMES.keys())
    default_idx = theme_keys.index(st.session_state.get('theme', theme_keys[0])) if st.session_state.get('theme') in theme_keys else 0
    st.radio('테마 선택', theme_keys, index=default_idx, key='theme', horizontal=True)
    sel = st.session_state.get('theme', theme_keys[0])
    pal = THEMES[sel]['palette']
    swatch_html = '<div style="display:flex;gap:6px;margin-top:8px;align-items:center">'
    for k in ['bg','card_bg','text','accent']:
        if k in pal:
            swatch_html += f"<div style=\"width:36px;height:24px;border-radius:6px;background:{pal[k]};border:1px solid rgba(0,0,0,0.06)\" title=\"{k}\"></div>"
    swatch_html += '</div>'
    st.markdown(swatch_html, unsafe_allow_html=True)
    st.write(THEMES[sel].get('desc',''))

with rescan_tab:
    st.markdown("**재스캔 워크플로**")
    group_list = sorted(list(df["그룹ID"].replace('-', pd.NA).dropna().unique())) if "그룹ID" in df.columns else []
    st.selectbox(
        "그룹 선택",
        ["전체"] + group_list,
        key="group_filter",
        help="재스캔 탭의 후보 목록을 특정 그룹으로 한정합니다.",
        on_change=switch_main_tab,
        args=("재스캔 필요",)
    )
    st.radio(
        "보기 방식",
        ["대형 비교(2열)", "그리드(다중 썸네일)"],
        key="group_view_mode",
        horizontal=True,
        help="대형 비교는 앞·뒤면을 크게 보여주고, 그리드는 그룹 내 모든 이미지를 타일로 확인합니다.",
        on_change=switch_main_tab,
        args=("재스캔 필요",)
    )
    rescan_quality_default = st.session_state.get("rescan_quality_profile", "균형")
    rescan_q_idx = quality_options_common.index(rescan_quality_default) if rescan_quality_default in quality_options_common else 1
    st.radio(
        "화질 프로파일",
        quality_options_common,
        index=rescan_q_idx,
        key="rescan_quality_profile",
        horizontal=True,
        help="빠름(512px), 균형(1024px), 선명(1600px) 수준으로 썸네일 품질과 크기를 조정합니다.",
        on_change=switch_main_tab,
        args=("재스캔 필요",)
    )

    delete_mode = st.session_state.get("rescan_delete_mode", False)
    delete_targets = st.session_state.get("rescan_delete_targets", [])
    waiting_confirm = st.session_state.get("rescan_show_confirm", False)

    if not delete_mode:
        delete_button_label = "🗑️ 삭제"
    else:
        if waiting_confirm:
            delete_button_label = "🗑️ 삭제 확인 중"
        elif delete_targets:
            delete_button_label = f"🗑️ 삭제 ({len(delete_targets)}개)"
        else:
            delete_button_label = "🗑️ 삭제 실행"

    if st.button(delete_button_label, key="rescan_delete_button"):
        if not delete_mode:
            st.session_state.rescan_delete_mode = True
            st.session_state.rescan_delete_targets = []
            st.session_state.rescan_show_confirm = False
            st.session_state.rescan_delete_feedback = None
        else:
            if delete_targets:
                st.session_state.rescan_show_confirm = True
            else:
                st.session_state.rescan_delete_feedback = ("warn", "삭제할 이미지를 먼저 선택하세요.")

    if delete_mode and not waiting_confirm:
        if st.button("취소", key="rescan_delete_cancel"):
            st.session_state.rescan_delete_mode = False
            st.session_state.rescan_delete_targets = []
            st.session_state.rescan_delete_feedback = None
            st.rerun()

with ok_tab:
    st.markdown("**정상/공백 답안 보기**")
    st.radio(
        "보기 옵션",
        ["모두 보기", "정상만", "공백만"],
        key="ok_view_mode",
        horizontal=True,
        help="정상/공백 탭에서 표시할 답안 유형을 빠르게 전환합니다.",
        on_change=switch_main_tab,
        args=("정상/공백 답안",)
    )
    grid_default = int(st.session_state.get("grid_cols", 5))
    grid_slider_args = {
        "label": "그리드 열 개수",
        "min_value": 2,
        "max_value": 10,
        "key": "grid_cols",
        "help": "정상/공백 탭의 썸네일 한 줄 배치를 조정합니다."
    }
    st.slider(value=grid_default, **grid_slider_args)
    ok_quality_default = st.session_state.get("ok_quality_profile", "균형")
    ok_q_idx = quality_options_common.index(ok_quality_default) if ok_quality_default in quality_options_common else 1
    st.radio(
        "화질 프로파일",
        quality_options_common,
        index=ok_q_idx,
        key="ok_quality_profile",
        horizontal=True,
        help="빠름(512px), 균형(1024px), 선명(1600px) 썸네일 품질을 선택합니다.",
        on_change=switch_main_tab,
        args=("정상/공백 답안",)
    )

with gallery_tab:
    st.markdown("**전체 보기 필터**")
    st.text_input(
        "파일명·경로 검색",
        key="gallery_search",
        placeholder="예: 10002, scan, .png",
        on_change=switch_main_tab,
        args=("전체 보기",)
    )
    ext_options = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    if "gallery_exts" in st.session_state:
        st.multiselect("확장자", ext_options, key="gallery_exts")
    else:
        st.multiselect("확장자", ext_options, default=[], key="gallery_exts")
    sort_options = ["파일명", "수정시각(최신순)", "수정시각(오래된순)"]
    sort_idx = sort_options.index(st.session_state.gallery_sort) if st.session_state.gallery_sort in sort_options else 0
    st.selectbox("정렬", sort_options, index=sort_idx, key="gallery_sort")
    st.markdown("---")
    st.markdown("**표시 설정**")
    quality_options = quality_options_common
    current_quality = st.session_state.get("gallery_quality_profile")
    q_idx = quality_options.index(current_quality) if current_quality in quality_options else 1
    st.radio(
        "화질 프로파일",
        quality_options,
        index=q_idx,
        horizontal=True,
        key="gallery_quality_profile",
        help="빠름(512px), 균형(1024px), 선명(1600px)",
        on_change=switch_main_tab,
        args=("전체 보기",)
    )
    render_options = ["리샘플(권장)", "원본"]
    r_idx = render_options.index(st.session_state.gallery_render_mode) if st.session_state.gallery_render_mode in render_options else 0
    st.radio(
        "렌더 방식",
        render_options,
        index=r_idx,
        horizontal=True,
        key="gallery_render_mode",
        help="리샘플: LANCZOS 고화질 썸네일 / 원본: 이미지 원본 로드",
        on_change=switch_main_tab,
        args=("전체 보기",)
    )
    gallery_grid_default = int(st.session_state.get("gallery_grid_cols", 5))
    gallery_slider_args = {
        "label": "그리드 열 개수",
        "min_value": 2,
        "max_value": 10,
        "key": "gallery_grid_cols",
        "help": "전체 보기 탭에서 한 줄에 배치될 썸네일 개수"
    }
    st.slider(value=gallery_grid_default, **gallery_slider_args)

_inject_theme_css(st.session_state.get('theme','Light (기본)'))

group_filter = st.session_state.get("group_filter", "전체")
grid_cols = int(st.session_state.get("grid_cols", 5))
group_view_mode = st.session_state.get("group_view_mode", "그리드(다중 썸네일)")
disp_fmt = "WEBP"    # 고정 포맷

rescan_quality_profile = st.session_state.get("rescan_quality_profile", "균형")
if rescan_quality_profile == "빠름":
    rescan_thumb_px, rescan_large_px, rescan_disp_quality = 512, 1200, 92
elif rescan_quality_profile == "선명":
    rescan_thumb_px, rescan_large_px, rescan_disp_quality = 1600, 2000, 98
else:  # 균형
    rescan_thumb_px, rescan_large_px, rescan_disp_quality = 1024, 1600, 95

ok_quality_profile = st.session_state.get("ok_quality_profile", "균형")
if ok_quality_profile == "빠름":
    ok_thumb_px, ok_disp_quality = 512, 92
elif ok_quality_profile == "선명":
    ok_thumb_px, ok_disp_quality = 1600, 98
else:
    ok_thumb_px, ok_disp_quality = 1024, 95

group_page_size = 6   # 고정값
group_page = 1        # 고정값(페이지네이션은 필요시만)

# ===== 유틸: 비교용 도구 =====
def _cv2_read_unicode(path: str, flag: int) -> Optional[np.ndarray]:
    """cv2.imread는 Windows에서 유니코드 경로를 처리하지 못할 수 있으므로 안전한 대안을 제공합니다."""
    if not path:
        return None
    arr = cv2.imread(path, flag)
    if arr is not None:
        return arr
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    try:
        return cv2.imdecode(data, flag)
    except Exception:
        return None


def _read_gray_same_size(a_path: str, b_path: str) -> Tuple[np.ndarray, np.ndarray]:
    a = _cv2_read_unicode(a_path, cv2.IMREAD_GRAYSCALE)
    b = _cv2_read_unicode(b_path, cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise RuntimeError(f"이미지 로드 실패: {a_path}")
    if b is None:
        raise RuntimeError(f"이미지 로드 실패: {b_path}")
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b

def _absdiff_heatmap(a: np.ndarray, b: np.ndarray, blur_size: int = 3, threshold: int = 10) -> np.ndarray:
    """두 그레이스케일 이미지의 절대차이를 heatmap으로 변환합니다.
    
    Args:
        a, b: 그레이스케일 이미지 배열
        blur_size: 가우시안 블러 크기 (홀수, 기본값 3)
        threshold: 차이 임계값 (기본값 10)
    
    Returns:
        RGB heatmap 배열
    """
    diff = cv2.absdiff(a, b)
    
    # 임계값 적용하여 노이즈 제거
    _, diff_thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # 블러 적용 (홀수만 허용)
    if blur_size > 1:
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        diff_thresh = cv2.GaussianBlur(diff_thresh, (blur_size, blur_size), 0)
    
    # 정규화 및 컬러맵 적용
    diff_norm = cv2.normalize(diff_thresh, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _blend_images_rgb(a_path: str, b_path: str, alpha: float = 0.5) -> np.ndarray:
    """두 이미지를 읽어 공통 최소 크기로 리사이즈한 뒤 RGB로 블렌드하여 numpy 배열을 반환합니다.
    alpha는 첫 번째 이미지(a)의 가중치(0..1)입니다."""
    a = _cv2_read_unicode(a_path, cv2.IMREAD_COLOR)
    b = _cv2_read_unicode(b_path, cv2.IMREAD_COLOR)
    if a is None:
        raise RuntimeError(f"이미지 로드 실패: {a_path}")
    if b is None:
        raise RuntimeError(f"이미지 로드 실패: {b_path}")
    # 최소 공통 크기로 리사이즈합니다
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    out = cv2.addWeighted(a.astype('float32'), alpha, b.astype('float32'), 1.0 - alpha, 0.0)
    out = out.astype('uint8')
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _highlight_differences_rgb(a_path: str, b_path: str, color: Tuple[int, int, int] = (0, 255, 255), thresh: int = 20) -> np.ndarray:
    """절대 차이가 thresh보다 큰 영역에 색 마스크를 오버레이해 변경점을 강조한 RGB 배열을 반환합니다.
    color는 OpenCV(BGR) 형식으로 전달하되, 반환값은 RGB입니다."""
    a = _cv2_read_unicode(a_path, cv2.IMREAD_COLOR)
    b = _cv2_read_unicode(b_path, cv2.IMREAD_COLOR)
    if a is None:
        raise RuntimeError(f"이미지 로드 실패: {a_path}")
    if b is None:
        raise RuntimeError(f"이미지 로드 실패: {b_path}")
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    # 노이즈를 줄이기 위해 약간의 블러를 적용합니다
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # color는 OpenCV의 BGR 형식을 기대합니다
    overlay = (mask3.astype('float32') / 255.0) * np.array(color, dtype='float32')
    base = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype('float32')
    # 결합: 마스크가 있는 곳에 오버레이 색상을 베이스와 혼합합니다
    alpha = 0.6
    combined = base * (1.0 - (mask3.astype('float32') / 255.0) * alpha) + overlay * alpha
    combined = np.clip(combined, 0, 255).astype('uint8')
    return combined


def _comp_cache_key(a_path: str, b_path: str, mode: str, params: Dict) -> str:
    # 비교 결과를 캐시하기 위한 고유 키(경로, 수정시간, 모드, 파라미터 포함)를 생성합니다
    s = f"{a_path}|{_file_mtime(a_path)}|{b_path}|{_file_mtime(b_path)}|{mode}|{sorted(params.items())}"
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def _write_cached_image(arr_rgb: np.ndarray, dst: str, fmt: str = 'PNG') -> Optional[str]:
    # numpy RGB 배열을 이미지 파일로 저장해 캐시에 보관합니다. 실패 시 None 반환
    try:
        img = Image.fromarray(arr_rgb)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img.save(dst, fmt)
        return dst
    except Exception as e:
        logger.warning(f"캐시 이미지 쓰기 실패 {dst}: {e}")
        return None


def _cached_blend_path(a_path: str, b_path: str, alpha: float = 0.5) -> Optional[str]:
    key = _comp_cache_key(a_path, b_path, 'fade', {'alpha': alpha})
    dst = os.path.join(THUMB_DIR, f"cmp_fade_{key}.png")
    if os.path.exists(dst) and _file_mtime(dst) >= max(_file_mtime(a_path), _file_mtime(b_path)):
        return dst
    arr = _blend_images_rgb(a_path, b_path, alpha=alpha)
    return _write_cached_image(arr, dst, fmt='PNG')


def _cached_heatmap_path(a_path: str, b_path: str, blur_size: int = 3, threshold: int = 10) -> Optional[str]:
    """차이 heatmap을 캐시에 생성하고 경로를 반환합니다."""
    key = _comp_cache_key(a_path, b_path, 'heatmap', {'blur': blur_size, 'thresh': threshold})
    dst = os.path.join(THUMB_DIR, f"cmp_heatmap_{key}.png")
    if os.path.exists(dst) and _file_mtime(dst) >= max(_file_mtime(a_path), _file_mtime(b_path)):
        return dst
    try:
        a_gray, b_gray = _read_gray_same_size(a_path, b_path)
        arr = _absdiff_heatmap(a_gray, b_gray, blur_size=blur_size, threshold=threshold)
        return _write_cached_image(arr, dst, fmt='PNG')
    except Exception as e:
        logger.warning(f"heatmap 캐시 생성 실패 {a_path} vs {b_path}: {e}")
        return None


def _cached_highlight_path(a_path: str, b_path: str, color: Tuple[int, int, int] = (0, 255, 255), thresh: int = 20) -> Optional[str]:
    key = _comp_cache_key(a_path, b_path, 'hl', {'color': color, 'thresh': thresh})
    dst = os.path.join(THUMB_DIR, f"cmp_hl_{key}.png")
    if os.path.exists(dst) and _file_mtime(dst) >= max(_file_mtime(a_path), _file_mtime(b_path)):
        return dst
    arr = _highlight_differences_rgb(a_path, b_path, color=color, thresh=thresh)
    return _write_cached_image(arr, dst, fmt='PNG')


# GIF 생성 지원 제거: Fade는 이제 _cached_blend_path/_blend_images_rgb의 정적 블렌드만 사용합니다

# ===== 공통: 리포트 필터링 =====
def filter_sort_report(_df: pd.DataFrame) -> pd.DataFrame:
    view = _df.copy()
    if group_filter != "전체" and "그룹ID" in view.columns:
        view = view[view["그룹ID"] == group_filter]
    # 기본 정렬: 유사도 내림차순, 그 다음 파일명
    if "유사도" in view.columns:
        view = view.sort_values(["유사도", "파일1", "파일2"], ascending=[False, True, True])
    else:
        view = view.sort_values(["파일1", "파일2"])
    return view

# ===== 세션: 비교 큐 =====
# 통합된 비교 선택 상태: 절대 경로 리스트 (최대 2개)
def toggle_compare(img_path: str):
    """비교를 위한 선택 토글 기능. 선택된 항목의 절대 경로를 $\text{gallery_selected}$에 저장합니다. (최대 2개)."""
    if not img_path:
        return
    # 가능한 경우 절대 경로로 정규화
    path = img_path
    if not os.path.isfile(path):
    # 베이스네임 맵으로 경로 해결을 시도
        resolved = resolve_image_path(path)
        if resolved and os.path.isfile(resolved):
            path = resolved
    # 이미 존재하면 선택 해제
    if path in st.session_state["gallery_selected"]:
        st.session_state["gallery_selected"] = [p for p in st.session_state["gallery_selected"] if p != path]
        return
    # 추가하되 최신 2개만 유지
    if len(st.session_state["gallery_selected"]) >= 2:
        st.session_state["gallery_selected"] = st.session_state["gallery_selected"][1:] + [path]
    else:
        st.session_state["gallery_selected"].append(path)


def toggle_delete_target(img_path: str):
    """재스캔 탭 삭제 모드에서 선택 대상을 토글합니다."""
    if not img_path:
        return
    targets = st.session_state.get("rescan_delete_targets", [])
    if img_path in targets:
        st.session_state.rescan_delete_targets = [p for p in targets if p != img_path]
    else:
        st.session_state.rescan_delete_targets = targets + [img_path]
    st.session_state.rescan_delete_feedback = None


def refresh_image_caches():
    """파일 삭제 후 이미지 관련 캐시와 경로 맵을 새로고침합니다."""
    global BASENAME_MAP
    try:
        list_all_images.clear()
    except AttributeError:
        pass
    try:
        build_basename_map.clear()
    except AttributeError:
        pass
    try:
        SEARCH_CACHE.clear()
    except Exception:
        pass
    try:
        BASENAME_MAP = build_basename_map(str(OUTPUT_DIR), cache_buster=time.time())
    except Exception as exc:
        logger.debug(f"BASENAME_MAP 갱신 실패: {exc}")


@st.cache_data(show_spinner=False)
def load_input_basename_map() -> Dict[str, List[str]]:
    """artifacts/ordered_paths.txt가 있으면 입력 폴더 경로들을 읽어
    베이스네임(소문자) -> 원본 절대 경로 목록으로 매핑합니다.
    중복 파일명은 모두 포함합니다.
    """
    mapping: Dict[str, List[str]] = {}
    try:
        ordered_txt = os.path.join(OUTPUT_DIR, "artifacts", "ordered_paths.txt")
        if not os.path.isfile(ordered_txt):
            return mapping
        with open(ordered_txt, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                p = line.strip().strip('\"')
                if not p:
                    continue
                # 경로가 실제 존재하는 경우만 포함
                if os.path.isabs(p) and os.path.exists(p):
                    bn = os.path.basename(p).lower()
                    mapping.setdefault(bn, []).append(p)
    except Exception as exc:
        logger.debug(f"입력 경로 맵 로드 실패: {exc}")
    return mapping


def _all_output_paths_by_basename(bn_lower: str) -> List[str]:
    """OUTPUT_DIR 하위에서 주어진 베이스네임과 일치하는 모든 파일 경로를 찾아 반환합니다."""
    try:
        # images_summary 또는 파일시스템에서 전체 목록을 가져온 뒤 필터
        all_imgs = list_all_images(OUTPUT_DIR, cache_buster=time.time())
        root_abs = os.path.abspath(str(OUTPUT_DIR))
        results = []
        for p in all_imgs:
            if os.path.basename(p).lower() != bn_lower:
                continue
            try:
                if os.path.commonpath([root_abs, os.path.abspath(p)]) == root_abs:
                    results.append(p)
            except Exception:
                continue
        return results
    except Exception:
        # 폴백: 실패 시 빈 목록 반환 (기본 경로 스캔에 의존)
        return []


def _remove_file_force(path: str) -> Optional[str]:
    """파일 삭제를 시도합니다. 읽기 전용/권한 문제를 처리하며, 성공 시 None, 실패 시 에러메시지 반환."""
    try:
        if not os.path.isfile(path):
            return "파일이 존재하지 않습니다."
        try:
            os.remove(path)
            return None
        except PermissionError:
            try:
                # 읽기 전용 해제 후 재시도 (Windows 대응)
                os.chmod(path, stat.S_IWRITE)
                os.remove(path)
                return None
            except Exception as exc:
                return f"권한 문제: {exc}"
        except Exception as exc:
            return str(exc)
    except Exception as exc:
        return str(exc)


def _clear_thumbnail_caches():
    """썸네일/비교 이미지 캐시를 깨끗이 비웁니다."""
    try:
        if os.path.isdir(THUMB_DIR):
            shutil.rmtree(THUMB_DIR, ignore_errors=True)
    except Exception as exc:
        logger.debug(f"썸네일 캐시 삭제 실패: {exc}")
    try:
        os.makedirs(os.path.join(THUMB_DIR, "disp_cache"), exist_ok=True)
    except Exception:
        pass


def delete_selected_images(target_paths: List[str], also_delete_input: bool = False) -> Tuple[List[str], List[Tuple[str, str]]]:
    """선택된 경로들을 기준으로 다음을 삭제합니다.
    - OUTPUT_DIR 하위의 동일 베이스네임 파일들(예: grouped/, ok/, blank_answers/ 등)
    - (옵션) artifacts/ordered_paths.txt에 기록된 원본 입력 폴더의 동일 베이스네임 파일들

    Returns:
        (성공 목록, 실패 (경로, 사유) 목록)
    """
    successes: List[str] = []
    failures: List[Tuple[str, str]] = []

    # 기준 베이스네임 집합 구성
    base_names = set()
    for p in target_paths:
        if p:
            base_names.add(os.path.basename(p).lower())

    # OUTPUT에서 모든 매칭 파일 수집
    to_delete: List[str] = []
    for bn in base_names:
        to_delete.extend(_all_output_paths_by_basename(bn))

    # 원본 입력 경로 매칭 (선택적)
    if also_delete_input:
        input_map = load_input_basename_map()
        for bn in base_names:
            for src in input_map.get(bn, []):
                to_delete.append(src)

    # 중복 제거 및 존재 확인
    unique_delete = []
    seen = set()
    for p in to_delete:
        if not p or p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            unique_delete.append(p)

    # 실제 삭제 수행
    for p in unique_delete:
        err = _remove_file_force(p)
        if err is None:
            successes.append(p)
        else:
            failures.append((p, err))

    # 캐시 및 맵 갱신
    _clear_thumbnail_caches()
    refresh_image_caches()

    return successes, failures


# 이미지를 base64로 인코딩하거나 카드 스타일을 생성하는 헬퍼들은
# 대시보드 UI의 특정 동적 기능을 위해 존재했습니다. 소규모 파이프라인용으로는
# 이러한 커스텀 CSS/인코딩 도우미를 제거하여 의존성/복잡도를 낮춥니다.

# @st.cache_data(show_spinner=False)
# def _encode_image_base64(img_path: str) -> str:
#     with open(img_path, "rb") as fh:
#         return base64.b64encode(fh.read()).decode("utf-8")
#
# def _delete_card_css(button_key: str, img_base64: str, selected: bool, height: int, disabled: bool) -> str:
#     # (삭제) 카드 스타일 생성용 복잡한 CSS를 여기서 생성하던 코드입니다.
#     # 소규모 UI에서는 기본 Streamlit 버튼/이미지 구성만으로 충분하다고 판단하여 제거했습니다.
#     return ""
# (삭제됨) 이전의 _delete_card_css에서 생성하던 복잡한 CSS 블록을 제거했습니다.
# 소규모 대시보드에서는 Streamlit의 기본 마크업/스타일로 충분하므로
# 사용자 정의 CSS를 대폭 줄였습니다.


def render_rescan_image_card(img_path: str, caption: str, key_suffix: str, target_px: int, quality: int, card_height: int) -> None:
    """재스캔 탭에서 이미지 카드를 렌더링합니다.
    삭제 모드일 때는 클릭 가능한 선택 카드로 표시하고,
    일반 모드일 때는 기본 이미지로 표시합니다.
    """
    if not img_path or not os.path.isfile(img_path):
        st.warning(f"이미지 파일을 찾을 수 없습니다: {caption}")
        return

    display_path = make_display_image(img_path, size=target_px, fmt=disp_fmt, quality=quality)
    delete_mode = st.session_state.get("rescan_delete_mode", False)
    waiting_confirm = st.session_state.get("rescan_show_confirm", False)
    selected = img_path in st.session_state.get("rescan_delete_targets", [])

    if delete_mode:
        # 삭제 모드: 이미지를 보여주고 선택 상태를 테두리로 표시
        container_key = f"img_container_{key_suffix}"
        button_key = f"select_btn_{key_suffix}"
        
        # 선택 상태에 따른 스타일
        border_style = "border: 4px solid #ef4444; box-shadow: 0 0 0 6px rgba(239,68,68,0.35);" if selected else "border: 2px solid rgba(148,163,184,0.45);"
        badge_text = "✓ 삭제 대상" if selected else "클릭하여 선택"
        badge_color = "background: rgba(239,68,68,0.95);" if selected else "background: rgba(15,23,42,0.65);"
        
        # 이미지와 선택 버튼을 함께 표시
        st.markdown(f"""
        <style>
        .img-select-container-{key_suffix} {{
            position: relative;
            {border_style}
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s ease;
        }}
        .img-select-container-{key_suffix}:hover {{
            transform: translateY(-2px);
            border-color: #ef4444;
        }}
        .img-select-badge-{key_suffix} {{
            position: absolute;
            bottom: 10px;
            right: 10px;
            {badge_color}
            color: white;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            z-index: 10;
            pointer-events: none;
        }}
        </style>
        <div class="img-select-container-{key_suffix}">
        """, unsafe_allow_html=True)
        
        # 이미지 표시
        st.image(_safe_image_open(display_path), use_container_width=True)
        
        st.markdown(f'<div class="img-select-badge-{key_suffix}">{badge_text}</div></div>', unsafe_allow_html=True)
        
        # 선택 버튼
        if st.button("🗑️ 선택" if not selected else "✓ 선택됨", 
                    key=button_key,
                    disabled=waiting_confirm,
                    on_click=toggle_delete_target,
                    args=(img_path,),
                    type="primary" if selected else "secondary",
                    use_container_width=True):
            pass
        
        st.caption(caption)
    else:
        st.image(_safe_image_open(display_path), caption=caption, use_container_width=True)

def on_image_click(img_path: str):
    """이미지를 클릭했을 때 비교 선택을 처리합니다."""
    toggle_compare(img_path)

# 이미지 클릭 이벤트를 처리하는 UI 요소에 on_image_click 함수를 연결합니다.
# 예를 들어, Streamlit의 st.image()를 사용할 경우:
# st.image(image_path, on_click=on_image_click, args=(image_path,))

# ===== 메인 탭 구성 (커스텀 탭 버튼으로 완전 제어) =====
tab_names = ["재스캔 필요", "정상/공백 답안", "전체 보기"]

# 탭 스타일 CSS
tab_css = """
<style>
.custom-tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 20px;
    border-bottom: 2px solid #e6e9ee;
    padding-bottom: 0;
}
.custom-tab {
    padding: 12px 24px;
    background: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    cursor: pointer;
    font-size: 16px;
    font-weight: 500;
    color: #666;
    transition: all 0.3s ease;
    margin-bottom: -2px;
}
.custom-tab:hover {
    color: #0B66FF;
    background: rgba(11, 102, 255, 0.05);
}
.custom-tab.active {
    color: #0B66FF;
    border-bottom-color: #0B66FF;
    font-weight: 600;
}
</style>
"""
st.markdown(tab_css, unsafe_allow_html=True)

# 탭 버튼 UI
cols = st.columns(len(tab_names))
for idx, tab_name in enumerate(tab_names):
    with cols[idx]:
        is_active = st.session_state["main_tab"] == tab_name
        button_type = "primary" if is_active else "secondary"
        if st.button(
            tab_name,
            key=f"tab_btn_{idx}",
            use_container_width=True,
            type=button_type
        ):
            st.session_state["main_tab"] = tab_name
            st.rerun()

st.markdown("---")

# ===== Global: 탭 어디에서든 2장 선택 시 상단에 즉시 비교 패널 표시 =====
def _render_global_compare():
    # gallery_selected는 절대 경로 리스트로 통일되어야 함
    paths = st.session_state.get('gallery_selected', [])
    sel_paths = [p for p in paths if p and os.path.isfile(p)]
    if len(sel_paths) >= 2:
        return sel_paths[:2]
    return None


cmp_pair = _render_global_compare()

    



# Tab1(리포트 요약) 관련 UI 블록은 현재 사용되지 않아 제거했습니다.
# 필요하면 향후 탭을 다시 추가하여 활성화할 수 있습니다.


# === Tab: 재스캔 필요 ===
if st.session_state["main_tab"] == "재스캔 필요":
    delete_mode = st.session_state.get("rescan_delete_mode", False)
    delete_targets = st.session_state.get("rescan_delete_targets", [])
    waiting_confirm = st.session_state.get("rescan_show_confirm", False)

    feedback = st.session_state.get("rescan_delete_feedback")
    if feedback:
        level, message = feedback
        if level == "success":
            st.success(message)
        elif level == "error":
            st.error(message)
        else:
            st.warning(message)
        st.session_state.rescan_delete_feedback = None

    if delete_mode and not waiting_confirm:
        if delete_targets:
            st.info(f"삭제 대상 {len(delete_targets)}개 선택됨: {', '.join(os.path.basename(p) for p in delete_targets)}")
        else:
            st.info("삭제할 이미지를 선택하세요. 이미지 아래의 '🗑️ 선택' 버튼을 눌러 토글할 수 있습니다.")

    if waiting_confirm:
        st.warning("선택한 이미지를 삭제하시겠습니까?")
        if delete_targets:
            grid_cols_confirm = min(4, max(1, len(delete_targets)))
            confirm_grid = st.columns(grid_cols_confirm)
            for idx, pth in enumerate(delete_targets):
                with confirm_grid[idx % grid_cols_confirm]:
                    if pth and os.path.isfile(pth):
                        # 최종 확인 단계에서는 썸네일 대신 고해상도 미리보기 사용
                        confirm_preview = make_display_image(
                            pth,
                            size=max(rescan_large_px, 1400),
                            fmt=disp_fmt,
                            quality=rescan_disp_quality,
                        )
                        st.image(_safe_image_open(confirm_preview), caption=os.path.basename(pth), use_container_width=True)
                    else:
                        st.info(f"파일을 찾을 수 없음: {os.path.basename(pth) if pth else '알 수 없음'}")
        # 옵션: 원본 입력 폴더에서도 같은 파일명을 삭제
        also_del_input = st.checkbox("입력 폴더에서도 같은 이름의 파일 삭제", value=False, help="파이프라인 입력으로 사용된 원본 폴더(artifacts/ordered_paths.txt 기준)에서도 동일한 파일명을 찾아 함께 삭제합니다.")
        confirm_cols = st.columns([1, 1, 6])
        with confirm_cols[0]:
            if st.button("네, 삭제합니다", key="rescan_delete_confirm_yes"):
                successes, failures = delete_selected_images(delete_targets, also_delete_input=also_del_input)

                # 성공한 경우 비교 선택 상태에서 제거합니다.
                if successes and "gallery_selected" in st.session_state:
                    st.session_state.gallery_selected = [p for p in st.session_state.gallery_selected if p not in successes]

                if failures and successes:
                    msg = "일부 파일만 삭제되었습니다: " + ", ".join(os.path.basename(p) for p, _ in failures)
                    st.session_state.rescan_delete_feedback = ("error", msg)
                elif failures and not successes:
                    detail = "; ".join(f"{os.path.basename(p)}: {err}" for p, err in failures)
                    st.session_state.rescan_delete_feedback = ("error", f"삭제 실패: {detail}")
                elif successes:
                    st.session_state.rescan_delete_feedback = ("success", f"{len(successes)}개 파일을 삭제했습니다.")
                else:
                    st.session_state.rescan_delete_feedback = ("warn", "삭제할 파일이 없습니다.")

                st.session_state.rescan_delete_targets = []
                st.session_state.rescan_delete_mode = False
                st.session_state.rescan_show_confirm = False

                _request_rerun()

        with confirm_cols[1]:
            if st.button("취소", key="rescan_delete_confirm_no"):
                st.session_state.rescan_show_confirm = False
                st.session_state.rescan_delete_mode = False
                st.session_state.rescan_delete_targets = []
                st.session_state.rescan_delete_feedback = None

    try:
        import imagehash
        dup_pairs = []
        hashes = {}
        # 현재는 사용자가 선택한 폴더에서 직접 스캔하지 않고 
        # 출력 폴더의 결과만을 기반으로 분석합니다
        scan_files = []  # 빈 리스트로 유지 (직접 스캔 없음)
        for f in scan_files:
            # 이 루프는 실행되지 않습니다 (scan_files가 빈 리스트이므로)
            # p = os.path.join(OUTPUT_DIR, f)  # 참조용 주석
            try:
                # 이 코드는 scan_files가 비어있어서 실행되지 않습니다
                # 안전하게 열기 (읽기 실패 파일은 건너뜀)
                # with open(p, 'rb') as fh:
                #     img = Image.open(fh).convert('L')
                #     h = imagehash.phash(img)
                # hashes[f] = h
                pass
            except Exception:
                # 읽기 실패 파일은 무시
                continue
        fl = list(hashes.keys())
        for i in range(len(fl)):
            for j in range(i+1, len(fl)):
                a, b = fl[i], fl[j]
                try:
                    d = abs(hashes[a] - hashes[b])
                except Exception:
                    continue
                # 매우 작은 해시 차이면 동일한 뒷장이 두 번 스캔된 것일 가능성 높음
                if d <= 1:
                    dup_pairs.append((a, b, int(d)))
    except Exception:
        dup_pairs = []

    # 재스캔 후보 생성: pHash 중복 쌍(input 폴더)과 리포트의 '유사 후보' 쌍을 합칩니다
    report_dups = []
    try:
        # df에는 파일명(리포트상)이 들어있음; 복원 가능한 경로로 변환
        if '상태' in df.columns:
            rpt = df[df['상태'] == '유사 후보'] if isinstance(df, pd.DataFrame) else pd.DataFrame()
            for _, row in rpt.iterrows():
                a, b = str(row.get('파일1', '')), str(row.get('파일2', ''))
                try:
                    sim = float(row.get('유사도', 0.0))
                except Exception:
                    sim = float(row.get('유사도', 0.0) or 0.0)
                pa = resolve_image_path(a) or os.path.join(OUTPUT_DIR, a)
                pb = resolve_image_path(b) or os.path.join(OUTPUT_DIR, b)
                report_dups.append((pa, pb, sim))
    except Exception:
        report_dups = []

    # 후보 병합 및 우선순위: pHash 쌍은 d가 작을수록 강한 신호이며, 리포트 쌍은 유사도(sim)가 클수록 우선
    candidates = []  # (score, a_path, b_path, source, meta) 튜플 목록
    try:
        for a, b, d in dup_pairs:
            pa = resolve_image_path(a) or os.path.join(OUTPUT_DIR, a)
            pb = resolve_image_path(b) or os.path.join(OUTPUT_DIR, b)
            # 점수: d가 작을수록 우선도 높음 -> 1/(1+d)로 반전
            score = 1.0 / (1.0 + float(d))
            candidates.append((score, pa, pb, 'phash', {'d': d}))
    except Exception:
        pass
    try:
        for pa, pb, sim in report_dups:
        # 리포트의 유사도는 이미 0..1 범위이므로 그대로 사용
            candidates.append((float(sim), pa, pb, 'report', {'sim': sim}))
    except Exception:
        pass

    # 점수 내림차순 정렬(값이 클수록 우선순위 높음)
    candidates = sorted([c for c in candidates if c[1] and c[2]], key=lambda x: x[0], reverse=True)

    # 베이스네임 쌍으로 중복 제거(순서 무시)
    seen = set()
    deduped = []
    for score, pa, pb, src, meta in candidates:
        key = tuple(sorted((os.path.basename(pa).lower(), os.path.basename(pb).lower())))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((score, pa, pb, src, meta))
    # ----- 즉시 비교 패널: 사용자가 아래 그리드에서 '↔ 비교 선택' 버튼을 클릭하면
    # rescan 탭의 상단에 바로 비교 옵션과 결과가 표시되도록 함
    # 통합된 gallery_selected(경로 리스트) 사용
    sel_exist_top = [p for p in st.session_state.get("gallery_selected", []) if p and os.path.isfile(p)]
    if sel_exist_top:
        st.markdown("---")
        st.markdown("### 🔍 즉시 비교 (재스캔 탭)")
        
        # 현재 선택된 이미지 정보 표시
        if len(sel_exist_top) == 1:
            st.info(f"📁 선택된 이미지: **{os.path.basename(sel_exist_top[0])}**")
        elif len(sel_exist_top) >= 2:
            a_name, b_name = os.path.basename(sel_exist_top[0]), os.path.basename(sel_exist_top[1])
            st.info(f"📁 비교 대상: **A**: {a_name} ↔ **B**: {b_name}")
            if len(sel_exist_top) > 2:
                st.caption(f" 추가로 {len(sel_exist_top)-2}개 이미지가 더 선택되어 있습니다. (최대 2개까지 비교)")
    
    # 공통: 모드 선택 + 도움말 옆에 배치
        colm1, colm2 = st.columns([3, 7])
        with colm1:
            # 내부 값(key)은 변경하지 않되, 사용자에게 보이는 라벨은 한국어로 제공합니다.
            cmp_mode_top = st.radio("보기 표시 (재스캔)", ["비교(좌우)","페이드(겹침)", "차이(Heatmap)", "하이라이터(오버레이)"], index=0, horizontal=True, key="cmp_mode_top")
        with colm2:
            # 선택된 비교 모드에 해당하는 설명만 표시
            # cmp_mode_top 내부값은 라디오의 label로 들어가므로 위젯의 라벨에 따라 분기합니다.
            if cmp_mode_top == "비교(좌우)":
                st.markdown(
                    "**비교(좌우)**\n"
                    "- 선택한 두 이미지를 좌우로 나란히 크게 보여줍니다. 빠르게 원본 대비를 확인할 때 사용하세요.\n"
                )
            elif cmp_mode_top == "페이드(겹침)":
                st.markdown(
                    "**페이드(겹침)**\n"
                    "- 두 이미지를 위아래로 겹쳐 보여줍니다. A 이미지의 투명도(alpha)를 조절해 미세한 변화 위치를 문맥과 함께 확인하세요.\n"
                    "- 권장: A alpha = 0.4–0.6\n"
                    "- 팁: 전체 레이아웃을 보존하므로 레이아웃 변화 식별에 유리합니다.\n"
                )
            elif cmp_mode_top == "차이(Heatmap)":
                st.markdown(
                    "**차이(Heatmap)**\n"
                    "- 그레이스케일 절대 차이를 계산해 heatmap으로 표시합니다. 픽셀 단위 변경을 강조합니다.\n"
                    "- 권장: Blur = 3, Threshold = 10\n"
                    "- 팁: 노이즈에 민감하므로 Blur/Threshold 조정으로 노이즈를 억제하세요.\n"
                )
            else:
                st.markdown(
                    "**하이라이터(오버레이)**\n"
                    "- 차이 마스크를 색상으로 원본 이미지에 오버레이합니다. 글자 추가/삭제 같은 의미 있는 변경을 빠르게 파악할 때 유용합니다.\n"
                    "- 권장: Threshold = 15, 색상 = Yellow\n"
                )

        # 모드별 파라미터 제어. '비교'는 추가 컨트롤이 없습니다.
        # 위젯이 렌더되지 않더라도 해당 변수가 존재하도록 보장(재실행 시 NameError 방지)
        # 기본값을 제공; 실제 위젯 선택 시 Streamlit이 세션 상태를 갱신합니다.
        if 'diff_blur_top' not in st.session_state:
            st.session_state['diff_blur_top'] = 3
        if 'diff_thresh_top' not in st.session_state:
            st.session_state['diff_thresh_top'] = 10
        if 'hl_color_top' not in st.session_state:
            st.session_state['hl_color_top'] = 'Yellow'
        if 'hl_thresh_top' not in st.session_state:
            st.session_state['hl_thresh_top'] = 20
        # 페이드(블렌드) 기본값
        if 'fade_alpha_top' not in st.session_state:
            st.session_state['fade_alpha_top'] = 0.5

        # 모드별 파라미터 위젯 (세션 상태를 갱신함)
        param_cols = st.columns([2, 2, 6])
        if cmp_mode_top == "비교(좌우)":
            # 단순 좌우 비교는 별도의 파라미터 없음
            pass
        elif cmp_mode_top == "페이드(겹침)":
            with param_cols[0]:
                fade_alpha_top = st.slider("A 이미지 알파", 0.0, 1.0, float(st.session_state.get('fade_alpha_top', 0.5)), 0.05, key='fade_alpha_top')
        elif cmp_mode_top == "차이(Heatmap)":
            with param_cols[0]:
                diff_blur_top = st.slider("블러", 1, 9, st.session_state.get('diff_blur_top', 3), 2, key='diff_blur_top')
            with param_cols[1]:
                diff_thresh_top = st.slider("임계값", 1, 50, st.session_state.get('diff_thresh_top', 10), 1, key='diff_thresh_top')
        elif cmp_mode_top == "하이라이터(오버레이)":
            with param_cols[0]:
                hl_color_top = st.selectbox("하이라이터 색상", ["Yellow", "Red", "Lime", "Cyan"], index=["Yellow", "Red", "Lime", "Cyan"].index(st.session_state.get('hl_color_top', 'Yellow')), key="hl_color_top")
            with param_cols[1]:
                hl_thresh_top = st.slider("임계값", 1, 100, st.session_state.get('hl_thresh_top', 20), 1, key="hl_thresh_top")

        if len(sel_exist_top) == 1:
            # 1개 선택 시에도 해제 버튼 제공
            clear_single_col = st.columns([1, 9])
            with clear_single_col[0]:
                if st.button("🗑️ 해제", key="clear_single_top", help="선택한 이미지 해제"):
                    st.session_state["gallery_selected"] = []
                    st.rerun()
            
            bigp = make_display_image(sel_exist_top[0], size=max(1400, rescan_large_px), fmt=disp_fmt, quality=rescan_disp_quality)
            st.image(_safe_image_open(bigp), caption=os.path.basename(sel_exist_top[0]), use_container_width=True)
        elif len(sel_exist_top) >= 2:
            a_path, b_path = sel_exist_top[:2]
            
            # 선택 해제 버튼들
            clear_cols = st.columns([1, 1, 1, 7])
            with clear_cols[0]:
                if st.button("🗑️ A 해제", key="clear_a_top", help="첫 번째 선택 이미지 해제"):
                    if a_path in st.session_state["gallery_selected"]:
                        st.session_state["gallery_selected"].remove(a_path)
                        st.rerun()
            with clear_cols[1]:
                if st.button("🗑️ B 해제", key="clear_b_top", help="두 번째 선택 이미지 해제"):
                    if b_path in st.session_state["gallery_selected"]:
                        st.session_state["gallery_selected"].remove(b_path)
                        st.rerun()
            with clear_cols[2]:
                if st.button("🗑️ 전체 해제", key="clear_all_top", help="모든 선택 이미지 해제"):
                    st.session_state["gallery_selected"] = []
                    st.rerun()
            
            # 사용자가 단순 비교(좌우)를 선택하면 두 이미지를 나란히 표시; 그렇지 않으면 병합/처리된 단일 이미지를 표시
            if cmp_mode_top == "비교(좌우)":
                big_a = make_display_image(a_path, size=max(1600, rescan_large_px), fmt=disp_fmt, quality=rescan_disp_quality)
                big_b = make_display_image(b_path, size=max(1600, rescan_large_px), fmt=disp_fmt, quality=rescan_disp_quality)
                c1t, c2t = st.columns(2)
                with c1t:
                    st.image(_safe_image_open(big_a), caption=f"A: {os.path.basename(a_path)}", use_container_width=True)
                with c2t:
                    st.image(_safe_image_open(big_b), caption=f"B: {os.path.basename(b_path)}", use_container_width=True)
            else:
                try:
                    if cmp_mode_top == "페이드(겹침)":
                        fade_alpha_val = float(st.session_state.get('fade_alpha_top', 0.5))
                        blendp = _cached_blend_path(a_path, b_path, alpha=fade_alpha_val)
                        if blendp and os.path.exists(blendp):
                            st.image(blendp, caption=f"페이드 블렌드 (A 알파={fade_alpha_val:.2f})", use_container_width=True)
                        else:
                            arr = _blend_images_rgb(a_path, b_path, alpha=fade_alpha_val)
                            st.image(arr, caption=f"페이드 블렌드 (A 알파={fade_alpha_val:.2f})", use_container_width=True)
                    elif cmp_mode_top == "차이(Heatmap)":
                        blur_val = st.session_state.get('diff_blur_top', 3)
                        thresh_val = st.session_state.get('diff_thresh_top', 10)
                        try:
                            # 캐시된 heatmap 사용
                            cached_heatmap = _cached_heatmap_path(a_path, b_path, blur_size=blur_val, threshold=thresh_val)
                            if cached_heatmap and os.path.exists(cached_heatmap):
                                st.image(cached_heatmap, caption=f"차이 Heatmap (블러={blur_val}, 임계={thresh_val})", use_container_width=True)
                            else:
                                # 캐시 실패 시 직접 생성
                                a_gray, b_gray = _read_gray_same_size(a_path, b_path)
                                heatmap = _absdiff_heatmap(a_gray, b_gray, blur_size=blur_val, threshold=thresh_val)
                                st.image(heatmap, caption=f"차이 Heatmap (블러={blur_val}, 임계={thresh_val})", use_container_width=True)
                        except Exception as he:
                            st.error(f"Heatmap 생성 실패: {he}")
                            # 폴백: 기본 좌우 비교
                            big_a = make_display_image(a_path, size=max(1600, rescan_large_px), fmt=disp_fmt, quality=rescan_disp_quality)
                            big_b = make_display_image(b_path, size=max(1600, rescan_large_px), fmt=disp_fmt, quality=rescan_disp_quality)
                            c1t, c2t = st.columns(2)
                            with c1t:
                                st.image(_safe_image_open(big_a), caption=f"A: {os.path.basename(a_path)}", use_container_width=True)
                            with c2t:
                                st.image(_safe_image_open(big_b), caption=f"B: {os.path.basename(b_path)}", use_container_width=True)
                    elif cmp_mode_top == "하이라이터(오버레이)":
                        color_map = {"Yellow": (0, 255, 255), "Red": (0, 0, 255), "Lime": (0, 255, 0), "Cyan": (255, 255, 0)}
                        # session_state에서 값을 읽되, NameError 방지를 위해 기본값을 사용
                        hl_color = st.session_state.get('hl_color_top', 'Yellow')
                        hl_thresh = st.session_state.get('hl_thresh_top', 20)
                        col_bgr = color_map.get(hl_color, (0, 255, 255))
                        cached = _cached_highlight_path(a_path, b_path, color=col_bgr, thresh=hl_thresh)
                        if cached and os.path.exists(cached):
                            st.image(cached, caption=f"하이라이터 오버레이 ({hl_color}, 임계={hl_thresh})", use_container_width=True)
                        else:
                            highlighted = _highlight_differences_rgb(a_path, b_path, color=col_bgr, thresh=hl_thresh)
                            st.image(highlighted, caption=f"하이라이터 오버레이 ({hl_color}, 임계={hl_thresh})", use_container_width=True)
                except Exception as e:
                    st.error(f"비교 렌더링 실패: {e}")
                    # 오류 발생 시 기본 좌우 비교로 폴백
                    st.info("기본 좌우 비교로 표시합니다.")
                    big_a = make_display_image(a_path, size=max(1600, rescan_large_px), fmt=disp_fmt, quality=rescan_disp_quality)
                    big_b = make_display_image(b_path, size=max(1600, rescan_large_px), fmt=disp_fmt, quality=rescan_disp_quality)
                    c1t, c2t = st.columns(2)
                    with c1t:
                        st.image(_safe_image_open(big_a), caption=f"A: {os.path.basename(a_path)}", use_container_width=True)
                    with c2t:
                        st.image(_safe_image_open(big_b), caption=f"B: {os.path.basename(b_path)}", use_container_width=True)

    # 파이프라인이 grouped/ok/blank_answers 폴더를 생성하지 않으므로
    # 파일시스템의 grouped 디렉터리 전용 로직을 제거했습니다.
    # 대신 리포트(df)와 images_summary(img_df)를 기반으로 후보를 렌더링합니다.
    # 그룹이 report 기반으로만 존재하는 경우 아래 블록이 처리합니다.
    group_rows = pd.DataFrame()
    groups: List[str] = []
    if report_available and isinstance(df, pd.DataFrame):
        required_cols = {"그룹ID", "파일1", "파일2"}
        if required_cols.issubset(df.columns):
            group_rows = df[df["그룹ID"].astype(str).str.strip().ne("-")]
            try:
                groups = sorted(group_rows["그룹ID"].astype(str).unique().tolist())
            except Exception:
                groups = []

        if group_filter != "전체":
            groups = [g for g in groups if g == group_filter]

    if not groups:
        st.info("표시할 재스캔 후보 그룹이 없습니다.")
    else:
        view_mode = st.session_state.get("group_view_mode", "그리드(다중 썸네일)")
        delete_mode = st.session_state.get("rescan_delete_mode", False)
        per_row = 2 if view_mode == "대형 비교(2열)" else 4
        display_px = rescan_large_px if view_mode == "대형 비교(2열)" else rescan_thumb_px
        card_height = 320 if view_mode == "대형 비교(2열)" else 220

        for gid in groups:
            rows = group_rows[group_rows["그룹ID"].astype(str) == gid]
            if rows.empty:
                continue

            st.markdown(f"### 그룹 {gid}")

            file_candidates: set[str] = set()
            for _, row in rows.iterrows():
                for col in ("파일1", "파일2"):
                    val = row.get(col)
                    if isinstance(val, str) and val:
                        file_candidates.add(val)

            back_files = [f for f in sorted(file_candidates) if is_2file(f)]
            if not back_files:
                st.caption("표시 가능한 이미지가 없습니다.")
                continue

            # 각 뒷면 파일 단위로 앞/뒤 이미지를 묶어 뷰 모드에 맞게 렌더링합니다.
            grouped_items: List[List[Dict[str, str]]] = []
            for back_name in back_files:
                back_path = resolve_image_path(back_name)
                front_name = corresponding_front_filename(back_name)
                front_path = resolve_image_path(front_name)

                entries: List[Dict[str, str]] = []
                if front_path and os.path.exists(front_path):
                    entries.append({"kind": "앞면", "name": front_name, "path": front_path})
                if back_path and os.path.exists(back_path):
                    entries.append({"kind": "뒷면", "name": back_name, "path": back_path})

                if entries:
                    grouped_items.append(entries)

            if not grouped_items:
                st.caption("표시 가능한 이미지가 없습니다.")
                continue

            def _render_tile(entry: Dict[str, str], pair_idx: int, item_idx: int) -> None:
                key_stub = f"{gid}_{pair_idx}_{item_idx}"
                label = f"{entry['kind']}: {entry['name']}"
                path = entry["path"]

                if delete_mode:
                    render_rescan_image_card(
                        path,
                        label,
                        key_stub,
                        display_px,
                        rescan_disp_quality,
                        card_height=card_height,
                    )
                    return

                selected = path in st.session_state.get("gallery_selected", [])
                button_label = "✔ 비교 취소" if selected else "↔ 비교 선택"
                button_key = f"cmp_rescan_{view_mode}_{key_stub}"
                if st.button(button_label, key=button_key):
                    toggle_compare(path)
                    st.rerun()

                disp = make_display_image(
                    path,
                    size=display_px,
                    fmt=disp_fmt,
                    quality=rescan_disp_quality,
                )
                st.image(
                    _safe_image_open(disp),
                    caption=label,
                    use_container_width=True,
                )

            if view_mode == "대형 비교(2열)":
                for pair_idx, bundle in enumerate(grouped_items):
                    cols = st.columns(min(per_row, len(bundle)))
                    for item_idx, entry in enumerate(bundle):
                        with cols[item_idx]:
                            _render_tile(entry, pair_idx, item_idx)
                st.markdown("")
            else:
                cols = st.columns(per_row)
                flat_entries = [
                    (entry, pair_idx, item_idx)
                    for pair_idx, bundle in enumerate(grouped_items)
                    for item_idx, entry in enumerate(bundle)
                ]
                for flat_idx, (entry, pair_idx, item_idx) in enumerate(flat_entries):
                    with cols[flat_idx % per_row]:
                        _render_tile(entry, pair_idx, item_idx)


# === Tab: 정상/공백 ===
elif st.session_state["main_tab"] == "정상/공백 답안":
    # 파일 시스템의 ok/blank_answers 하위 폴더에 의존하지 않고
    # images_summary(df)와 report(df)를 기반으로 정상/공백 뷰를 제공합니다.
    sel = st.session_state.get("ok_view_mode", "모두 보기")

    def _render_from_summary(title: str, mask: pd.Series) -> None:
        subset = img_df[mask] if isinstance(img_df, pd.DataFrame) else pd.DataFrame()
        st.subheader(title)
        if subset.empty or "파일" not in subset.columns:
            st.caption("표시할 이미지가 없습니다.")
            return
        files = [f for f in subset["파일"].astype(str).tolist() if is_2file(f) and _is_back_page(f)]
        if not files:
            st.caption("표시할 항목이 없습니다.")
            return
        cols = st.columns(grid_cols)
        for idx, f in enumerate(files):
            img_path = resolve_image_path(f)
            if not img_path or not os.path.exists(img_path):
                continue
            disp = make_display_image(img_path, size=ok_thumb_px, fmt=disp_fmt, quality=ok_disp_quality)
            with cols[idx % grid_cols]:
                st.image(_safe_image_open(disp), caption=f, use_container_width=True)

    if sel in ["모두 보기", "정상만"]:
        if "빈칸여부" in img_df.columns:
            mask = ~img_df["빈칸여부"].astype(bool)
        else:
            mask = pd.Series([False] * len(img_df), index=img_df.index)
        _render_from_summary("✅ 정상 답안", mask)

    if sel in ["모두 보기", "공백만"]:
        if "빈칸여부" in img_df.columns:
            mask = img_df["빈칸여부"].astype(bool)
        else:
            mask = pd.Series([False] * len(img_df), index=img_df.index)
        _render_from_summary("⭕ 공백 답안", mask)


# === Tab: 전체 보기 ===
elif st.session_state["main_tab"] == "전체 보기":
    quality_profile = st.session_state.get("gallery_quality_profile", "균형")
    render_mode = st.session_state.get("gallery_render_mode", "리샘플(권장)")
    grid_cols_local = max(2, int(st.session_state.get("gallery_grid_cols", 5)))
    q = st.session_state.get("gallery_search", "")
    ext_sel = st.session_state.get("gallery_exts", [])
    sort_key = st.session_state.get("gallery_sort", "파일명")

    if render_mode == "원본":
        st.caption("원본 모드: 대용량 이미지는 로딩 시간이 길어질 수 있습니다. 필요한 비교 구간에서만 사용하세요.")

    # 프로파일 → 표시 해상도/포맷/품질 파라미터 도출
    if quality_profile == "빠름":
        target_px_eff, disp_fmt_eff, disp_quality_eff = 512, "WEBP", 92
    elif quality_profile == "균형":
        target_px_eff, disp_fmt_eff, disp_quality_eff = 1024, "WEBP", 95
    else:  # 선명
        target_px_eff, disp_fmt_eff, disp_quality_eff = 1600, "WEBP", 98


    # ---------- 데이터 준비 ----------
    # # 보고서 업데이트 시 images_summary.csv 파일의 수정 시간(mtime)을 사용하여 캐시를 무효화(갱신)합니다.
    try:
        cache_buster_tuple = (
            os.path.getmtime(IMG_SUMMARY),
            len(os.listdir(os.path.dirname(IMG_SUMMARY)))
        )
    except Exception:
        cache_buster_tuple = (0, 0)
    all_imgs = list_all_images(OUTPUT_DIR, cache_buster=hash(cache_buster_tuple))
    # 검색/확장자 필터
    if q:
        all_imgs = [p for p in all_imgs if q.lower() in p.lower()]
    if ext_sel:
        all_imgs = [p for p in all_imgs if any(p.lower().endswith(e) for e in ext_sel)]

    # 정렬
    if sort_key == "파일명":
        all_imgs.sort(key=lambda p: os.path.basename(p).lower())
    elif sort_key == "수정시각(최신순)":
        all_imgs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    else:
        all_imgs.sort(key=lambda p: os.path.getmtime(p))

    total_items = len(all_imgs)
    st.caption(f"총 {total_items}개 파일")

    # 페이징/지연 로드: gallery_limit만큼만 표시. '더 보기'로 증분.
    if 'gallery_limit' not in st.session_state or st.session_state.gallery_limit <= 0:
        st.session_state.gallery_limit = 60
    # cap to total
    st.session_state.gallery_limit = min(total_items, st.session_state.gallery_limit)
    show_paths = all_imgs[:st.session_state.gallery_limit]
    st.caption(f"1–{len(show_paths)} / {total_items}")

    # ========== 이미지 그리드 섹션 ==========
    cols = st.columns(grid_cols_local)
    for idx, path in enumerate(show_paths):
        # 표시에 사용할 이미지(리샘플 or 원본)
        if render_mode == "원본":
            disp = path
        else:
            disp = make_display_image(path, size=target_px_eff, fmt=disp_fmt_eff, quality=disp_quality_eff)

        with cols[idx % grid_cols_local]:
            # 이미지 (클릭하면 비교 선택 토글)
            img_name = os.path.basename(path)
            try:
                st.image(
                    _safe_image_open(disp),
                    caption=img_name,
                    use_container_width=True,
                    key=f"gallery_img_{idx}",
                    on_click=on_image_click,
                    args=(path,),
                )
            except Exception:
                # 일부 streamlit 버전에서는 st.image가 on_click을 지원하지 않을 수 있으므로
                # 실패하면 폴백으로 클릭 없는 이미지를 표시합니다.
                st.image(_safe_image_open(disp), caption=img_name, use_container_width=True)

    # ---------- 더 보기 버튼 ----------
    if st.session_state.gallery_limit < total_items:
        if st.button("더 보기"): 
            # 한 번에 60장씩 추가
            st.session_state.gallery_limit = min(total_items, st.session_state.gallery_limit + 60)
            _request_rerun()

    # 우선 비교 토글로 대체 — 모달형 미리보기 버튼 제거

    # 모달 미지원 대체 표시: 없음(직접 inline으로 대체됨)