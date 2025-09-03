import os
import glob
from typing import Tuple

import streamlit as st
import polars as pl
from PIL import Image
import numpy as np
import pandas as pd

# Diff & metrics
import cv2
try:
    from skimage.metrics import structural_similarity as ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# Optional components
_HAS_IMG_CMP = False
try:
    from streamlit_image_comparison import image_comparison
    _HAS_IMG_CMP = True
except Exception:
    pass

_HAS_IMG_ZOOM = False
try:
    from streamlit_image_zoom import image_zoom
    _HAS_IMG_ZOOM = True
except Exception:
    pass

# ---------------- Args ----------------
# streamlit run src/dashboard.py -- --output_dir=output
import argparse
import sys


def parse_streamlit_args():
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        user_args = sys.argv[idx + 1:]
    else:
        user_args = []
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--output_dir', default='output')
    try:
        ns, _ = p.parse_known_args(user_args)
    except SystemExit:
        class X:
            output_dir = 'output'
        ns = X()
    return ns


ns = parse_streamlit_args()
OUTPUT_DIR = ns.output_dir
REPORT_PARQUET = os.path.join(OUTPUT_DIR, "report.parquet")
REPORT_CSV = os.path.join(OUTPUT_DIR, "report.csv")
IMG_SUMMARY = os.path.join(OUTPUT_DIR, "images_summary.csv")

st.set_page_config(page_title="답안지 검수 대시보드", layout="wide")
st.title("📋 답안지 스캔 검수 대시보드")

# ---------------- 데이터 로딩 ----------------
@st.cache_data(show_spinner=False)
def load_report():
    if os.path.exists(REPORT_PARQUET):
        df = pl.read_parquet(REPORT_PARQUET).to_pandas()
    elif os.path.exists(REPORT_CSV):
        df = pl.read_csv(REPORT_CSV).to_pandas()
    else:
        st.error("⚠️ 결과 파일이 없습니다. 먼저 main.py를 실행하세요.")
        st.stop()
    return df

@st.cache_data(show_spinner=False)
def load_img_summary():
    if os.path.exists(IMG_SUMMARY):
        return pd.read_csv(IMG_SUMMARY)
    return pd.DataFrame(columns=["파일", "밀도", "빈칸여부"])


df = load_report()
img_df = load_img_summary()

# ---------------- 세션 상태 ----------------
if "compare_list" not in st.session_state:
    st.session_state.compare_list = []


def toggle_compare(img_path: str):
    if img_path not in st.session_state.compare_list:
        st.session_state.compare_list.append(img_path)
    if len(st.session_state.compare_list) > 2:
        st.session_state.compare_list = st.session_state.compare_list[-2:]


# ---------------- Sidebar Filters ----------------
st.sidebar.header("필터 & 설정")
min_sim = st.sidebar.slider("최소 유사도 필터", 0.0, 1.0, 0.95, 0.001)
show_suspects = st.sidebar.checkbox("유사 후보 포함", value=True)

def _status_ok(s):
    if s == "중복/그룹":
        return True
    return show_suspects and (s == "유사 후보")


# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["리포트 요약", "유사 그룹", "정상/공백 답안", "전체 보기"])

# 📊 리포트 요약
with tab1:
    st.header("리포트 요약")
    if len(df):
        df_view = df[(df["유사도"] >= min_sim) & (df["상태"].apply(_status_ok))]
    else:
        df_view = df
    st.dataframe(df_view, use_container_width=True)
    st.download_button("⬇ CSV 다운로드", df.to_csv(index=False).encode("utf-8-sig"), "report.csv", "text/csv")
    try:
        n_groups = df['그룹ID'].replace('-', pd.NA).dropna().nunique()
    except Exception:
        n_groups = 0
    st.write(f"총 그룹 수: {n_groups}")

# 🖼️ 유사 그룹
with tab2:
    st.header("유사 그룹 보기")
    grouped_dir = os.path.join(OUTPUT_DIR, "grouped")
    if os.path.isdir(grouped_dir):
        groups = sorted(os.listdir(grouped_dir))
        sel = st.selectbox("그룹 선택", ["전체 그룹 보기"] + groups)
        targets = groups if sel == "전체 그룹 보기" else [sel]

        for gid in targets:
            st.subheader(f"그룹: {gid}")
            files = sorted(os.listdir(os.path.join(grouped_dir, gid)))
            cols = st.columns(4)

            for idx, f in enumerate(files):
                img_path = os.path.join(grouped_dir, gid, f)
                with cols[idx % 4]:
                    if st.button(f"선택 {f}", key=f"select_{gid}_{idx}"):
                        toggle_compare(img_path)
                    if _HAS_IMG_ZOOM:
                        image_zoom(img_path, width=300)
                    else:
                        st.image(Image.open(img_path), caption=f, use_container_width=True)

        # 선택된 이미지가 2개면 즉시 좌우 비교
        if len(st.session_state.compare_list) == 2:
            img1, img2 = st.session_state.compare_list
            st.markdown("### 🔍 선택한 이미지 비교")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**왼쪽**")
                if _HAS_IMG_ZOOM:
                    image_zoom(img1)
                else:
                    st.image(Image.open(img1), caption=os.path.basename(img1), use_container_width=True)
            with col2:
                st.markdown("**오른쪽**")
                if _HAS_IMG_ZOOM:
                    image_zoom(img2)
                else:
                    st.image(Image.open(img2), caption=os.path.basename(img2), use_container_width=True)

            # Slider comparison (있으면 사용)
            if _HAS_IMG_CMP:
                st.markdown("#### Slider 비교")
                image_comparison(
                    img1, img2,
                    label1=os.path.basename(img1), label2=os.path.basename(img2),
                    width=700
                )

            # Diff/SSIM heatmaps — 공백/연한 획도 잘 보이도록 강조
            st.markdown("#### 차이(AbsDiff) & SSIM 맵")
            diff_col1, diff_col2 = st.columns(2)
            with diff_col1:
                st.image(_absdiff_heatmap(img1, img2), caption="차이 강조(AbsDiff Heatmap)", use_container_width=True)
            with diff_col2:
                if _HAS_SKIMAGE:
                    ssim_img, ssim_score = _ssim_map(img1, img2)
                    st.image(ssim_img, caption=f"SSIM 맵 (mean={ssim_score:.4f})", use_container_width=True)
                else:
                    st.info("scikit-image 미설치로 SSIM 맵 생략")

            # 비교 후 초기화 (유지하고 싶으면 주석 처리)
            st.session_state.compare_list = []
    else:
        st.info("그룹 결과 폴더가 없습니다. 먼저 파이프라인을 실행하세요.")

# ✅ 정상/공백
with tab3:
    st.header("정상 / 공백 답안 보기")
    ok_dir = os.path.join(OUTPUT_DIR, "ok")
    blank_dir = os.path.join(OUTPUT_DIR, "blank_answers")
    sel = st.radio("보기 옵션", ["모두 보기", "정상만", "공백만"], horizontal=True)

    if sel in ["모두 보기", "정상만"] and os.path.isdir(ok_dir):
        st.subheader("✅ 정상 답안")
        files = sorted(os.listdir(ok_dir))
        cols = st.columns(5)
        for idx, f in enumerate(files):
            img_path = os.path.join(ok_dir, f)
            with cols[idx % 5]:
                st.image(Image.open(img_path), caption=f, use_container_width=True)

    if sel in ["모두 보기", "공백만"] and os.path.isdir(blank_dir):
        st.subheader("⭕ 공백 답안")
        files = sorted(os.listdir(blank_dir))
        cols = st.columns(5)
        for idx, f in enumerate(files):
            img_path = os.path.join(blank_dir, f)
            with cols[idx % 5]:
                st.image(Image.open(img_path), caption=f, use_container_width=True)

# 🌐 전체 보기
with tab4:
    st.header("전체 이미지 보기")
    all_imgs = glob.glob(os.path.join(OUTPUT_DIR, "**", "*.jpg"), recursive=True)
    all_imgs += glob.glob(os.path.join(OUTPUT_DIR, "**", "*.png"), recursive=True)
    cols = st.columns(5)
    for idx, path in enumerate(sorted(all_imgs)):
        with cols[idx % 5]:
            st.image(Image.open(path), caption=os.path.basename(path), use_container_width=True)


# ---------------- Utilities ----------------
def _read_gray_same_size(a_path: str, b_path: str) -> Tuple[np.ndarray, np.ndarray]:
    a = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        raise RuntimeError("이미지 로딩 실패")
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b


def _absdiff_heatmap(a_path: str, b_path: str) -> np.ndarray:
    a, b = _read_gray_same_size(a_path, b_path)
    diff = cv2.absdiff(a, b)
    # 미세한 간극(띄어쓰기/엷은 획) 강조: 살짝 블러 → 정규화 → 컬러맵
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    return heat[:, :, ::-1]  # BGR→RGB


def _ssim_map(a_path: str, b_path: str):
    a, b = _read_gray_same_size(a_path, b_path)
    score, ssim_img = ssim(a, b, full=True, data_range=255)
    ssim_img = (1.0 - ssim_img)  # 차이를 밝게
    ssim_img = (255 * (ssim_img / (ssim_img.max() + 1e-6))).astype(np.uint8)
    heat = cv2.applyColorMap(ssim_img, cv2.COLORMAP_INFERNO)
    return heat[:, :, ::-1], float(score)
