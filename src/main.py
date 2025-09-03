import os
import argparse
import subprocess
from detector_pipeline import detect_pipeline, DetectorConfig


def parse_args():
    p = argparse.ArgumentParser(description="Answer Sheet QA — run pipeline & dashboard")
    p.add_argument("--input_dir", default="input_images", help="Input images directory")
    p.add_argument("--output_dir", default="output", help="Output directory")
    p.add_argument("--use_faiss", action="store_true", help="Force FAISS if installed")
    p.add_argument("--k", type=int, default=20, help="KNN candidate neighbors per image")
    p.add_argument("--phash_thresh", type=int, default=10, help="Max Hamming distance for pHash prefilter")
    p.add_argument("--density_diff", type=float, default=0.15, help="Max absolute density difference prefilter")
    p.add_argument("--cnn_thresh", type=float, default=0.99, help="Similarity threshold for grouping")
    p.add_argument("--suspect_low", type=float, default=0.95, help="Similarity for suspect pairs")
    p.add_argument("--blank_thresh", type=float, default=0.02, help="Ink density <= this → blank answer")
    p.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0=disable multiproc)")
    p.add_argument("--roi", type=float, nargs=4, default=[0.15, 0.15, 0.85, 0.85],
                   help="ROI as ratios: left, top, right, bottom")
    return p.parse_args()


def main():
    args = parse_args()

    # Ensure output subdirectories exist
    output_dir = args.output_dir
    for sub in ["grouped", "ok", "blank_answers", "artifacts"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    print("🔍 탐지 실행 중... (pHash → FAISS/KNN → CNN)")
    config = DetectorConfig(
        use_faiss=args.use_faiss,
        k=args.k,
        phash_thresh=args.phash_thresh,
        density_diff_thresh=args.density_diff,
        cnn_thresh=args.cnn_thresh,
        suspect_low=args.suspect_low,
        blank_density_thresh=args.blank_thresh,
        batch_size=args.batch,
        num_workers=args.num_workers,
        roi_ratio=tuple(args.roi),
    )

    results, groups = detect_pipeline(args.input_dir, args.output_dir, config=config)
    print("✅ 탐지 완료 → report.csv, report.parquet 생성됨.")

    # 자동 대시보드 실행
    print("🌐 대시보드를 실행합니다... 브라우저에서 자동으로 열립니다.")
    subprocess.run([
        "python", "-m", "streamlit", "run", "src/dashboard.py", "--",
        f"--output_dir={args.output_dir}"
    ])


if __name__ == "__main__":
    main()
