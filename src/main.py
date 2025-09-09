import os
import argparse
import subprocess
from detector_pipeline import detect_pipeline, DetectorConfig

def parse_args():
    p = argparse.ArgumentParser(description="Answer Sheet QA — pipeline & dashboard (Handwriting-Optimized)")
    p.add_argument("--input_dir", default="input_images")
    p.add_argument("--output_dir", default="output")

    # Backends
    p.add_argument("--embed_backend", choices=["resnet18", "dinov2"], default="dinov2")
    p.add_argument("--ann_backend", choices=["auto", "brute", "faiss", "hnsw"], default="auto")

    # ANN params
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--hnsw_M", type=int, default=32)
    p.add_argument("--hnsw_efC", type=int, default=200)
    p.add_argument("--hnsw_efS", type=int, default=64)

    # Prefilters
    p.add_argument("--prefilter", choices=["phash", "pdq", "both"], default="phash")
    p.add_argument("--phash_thresh", type=int, default=10)
    p.add_argument("--pdq_thresh", type=int, default=80)
    p.add_argument("--density_diff", type=float, default=0.15)

    # Similarity thresholds
    p.add_argument("--cnn_thresh", type=float, default=0.99)
    p.add_argument("--suspect_low", type=float, default=0.95)

    # Blank detection
    p.add_argument("--blank_method", choices=["otsu", "sauvola"], default="sauvola")
    p.add_argument("--blank_thresh", type=float, default=0.02)

    # Re-ranking / OCR (optional)
    p.add_argument("--use_lpips", action="store_true")
    p.add_argument("--lpips_thresh", type=float, default=0.2)
    p.add_argument("--use_ocr", action="store_true")
    p.add_argument("--text_sim_thresh", type=float, default=0.85)

    # Alignment
    p.add_argument("--use_alignment", action="store_true")

    # Embedding
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--roi", type=float, nargs=4, default=[0.15, 0.15, 0.85, 0.85])
    p.add_argument("--detach", action="store_true", help="윈도우에서 Streamlit을 새 창으로 분리 실행합니다 (비차단).")
    return p.parse_args()

def main():
    args = parse_args()
    for sub in ["grouped", "ok", "blank_answers", "artifacts"]:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    print("🔍 탐지 실행…")
    cfg = DetectorConfig(
        embed_backend=args.embed_backend,
        ann_backend=args.ann_backend,
        k=args.k,   
        hnsw_M=args.hnsw_M,
        hnsw_efC=args.hnsw_efC,
        hnsw_efS=args.hnsw_efS,
        prefilter=args.prefilter,
        phash_thresh=args.phash_thresh,
        pdq_thresh=args.pdq_thresh,
        density_diff_thresh=args.density_diff,
        cnn_thresh=args.cnn_thresh,
        suspect_low=args.suspect_low,
        blank_method=args.blank_method,
        blank_density_thresh=args.blank_thresh,
        use_lpips=args.use_lpips,
        lpips_thresh=args.lpips_thresh,
        use_ocr=args.use_ocr,
        text_sim_thresh=args.text_sim_thresh,
        use_alignment=args.use_alignment,
        batch_size=args.batch,
        num_workers=args.num_workers,
        roi_ratio=tuple(args.roi),
    )

    detect_pipeline(args.input_dir, args.output_dir, config=cfg)
    print("✅ 완료 → report.csv, report.parquet, images_summary.csv 생성")

    print("🌐 대시보드 실행…")
    cmd = ["python", "-m", "streamlit", "run", "src/dashboard.py", "--", f"--output_dir={args.output_dir}"]
    try:
        if args.detach and os.name == 'nt':   
            subprocess.Popen(["cmd", "/c", "start"] + cmd)
        else:
            subprocess.run(cmd)
    except KeyboardInterrupt:
        print("중단: 사용자가 실행을 취소했습니다.")
    except Exception as e:
        print(f"대시보드 실행 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()