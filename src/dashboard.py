import os, glob
import streamlit as st
import pandas as pd
import polars as pl
from PIL import Image
from streamlit_image_comparison import image_comparison

OUTPUT_DIR = "output"
REPORT_PARQUET = os.path.join(OUTPUT_DIR, "report.parquet")
REPORT_CSV = os.path.join(OUTPUT_DIR, "report.csv")

st.set_page_config(page_title="답안지 검수 대시보드", layout="wide")
st.title("📋 답안지 스캔 검수 대시보드")

# ---------------- 데이터 로딩 ----------------
@st.cache_data
def load_report():
    if os.path.exists(REPORT_PARQUET):
        df = pl.read_parquet(REPORT_PARQUET)
    elif os.path.exists(REPORT_CSV):
        df = pl.read_csv(REPORT_CSV)
    else:
        st.error("⚠️ 결과 파일이 없습니다. 먼저 main.py를 실행하세요.")
        st.stop()
    return df.to_pandas()

df = load_report()

if "compare_list" not in st.session_state:
    st.session_state.compare_list = []

def toggle_compare(img_path):
    if img_path in st.session_state.compare_list:
        st.session_state.compare_list.remove(img_path)
    else:
        st.session_state.compare_list.append(img_path)
    if len(st.session_state.compare_list) > 2:
        st.session_state.compare_list = st.session_state.compare_list[-2:]

# ---------------- 탭 ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["리포트 요약", "유사 그룹", "정상/공백 답안", "이미지 비교", "전체 보기"]
)

# 📊 리포트 요약
with tab1:
    st.header("리포트 요약")
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇ CSV 다운로드",
        df.to_csv(index=False).encode("utf-8-sig"),
        "report.csv","text/csv")
    st.write(f"총 그룹 수: {df['그룹ID'].nunique() - (df['그룹ID'] == '-').sum()}")

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
                    if st.button(f"비교 선택 {f}", key=f"group_{gid}_{idx}"):
                        toggle_compare(img_path)
                    st.image(Image.open(img_path), caption=f, use_container_width=True)

# ✅ 정상/공백
with tab3:
    st.header("정상 / 공백 답안 보기")
    ok_dir = os.path.join(OUTPUT_DIR, "ok")
    blank_dir = os.path.join(OUTPUT_DIR, "blank_answers")
    sel = st.radio("보기 옵션", ["모두 보기","정상만","공백만"], horizontal=True)

    if sel in ["모두 보기","정상만"] and os.path.isdir(ok_dir):
        st.subheader("✅ 정상 답안")
        files = sorted(os.listdir(ok_dir))
        cols = st.columns(5)
        for idx, f in enumerate(files):
            img_path = os.path.join(ok_dir, f)
            with cols[idx % 5]:
                if st.button(f"비교 선택 {f}", key=f"ok_{idx}"):
                    toggle_compare(img_path)
                st.image(Image.open(img_path), caption=f, use_container_width=True)

    if sel in ["모두 보기","공백만"] and os.path.isdir(blank_dir):
        st.subheader("⭕ 공백 답안")
        files = sorted(os.listdir(blank_dir))
        cols = st.columns(5)
        for idx, f in enumerate(files):
            img_path = os.path.join(blank_dir, f)
            with cols[idx % 5]:
                if st.button(f"비교 선택 {f}", key=f"blank_{idx}"):
                    toggle_compare(img_path)
                st.image(Image.open(img_path), caption=f, use_container_width=True)

# 🔍 이미지 비교 (슬라이더)
with tab4:
    st.header("이미지 비교 (슬라이더)")
    if len(st.session_state.compare_list) == 2:
        img1, img2 = st.session_state.compare_list
        image_comparison(
            img1=img1, img2=img2,
            label1=os.path.basename(img1),
            label2=os.path.basename(img2),
            width=800, starting_position=50, show_labels=True
        )
    else:
        st.info("썸네일에서 비교할 이미지를 2개 선택하세요.")

# 🌐 전체 보기
with tab5:
    st.header("전체 이미지 보기")
    all_imgs = glob.glob(os.path.join(OUTPUT_DIR, "**", "*.jpg"), recursive=True)
    cols = st.columns(5)
    for idx, path in enumerate(sorted(all_imgs)):
        with cols[idx % 5]:
            if st.button(f"비교 선택 {os.path.basename(path)}", key=f"all_{idx}"):
                toggle_compare(path)
            st.image(Image.open(path), caption=os.path.basename(path), use_container_width=True)
