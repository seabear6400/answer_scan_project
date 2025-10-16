import argparse
import os
import sys
import time
import logging
from typing import Optional, Tuple

# OpenCV, timm 등의 불필요한 경고를 억제
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
logging.getLogger('timm').setLevel(logging.ERROR)


def parse_args():
    p = argparse.ArgumentParser(
        description="Answer Sheet QA — pipeline only (Handwriting-Optimized)"
    )
    # 입력/출력
    p.add_argument("--input_dir", help="분석할 이미지 폴더(미지정 시 GUI로 선택)")
    p.add_argument("--output_dir", default="output", help="결과 출력 폴더")
    p.add_argument("--recursive", action="store_true", help="하위 폴더까지 재귀적으로 이미지 검색")

    # 백엔드
    p.add_argument("--embed_backend", choices=["auto", "resnet18", "dinov2"], default="auto")
    p.add_argument("--ann_backend", choices=["auto", "brute", "faiss", "hnsw"], default="auto")

    # ANN 파라미터
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--hnsw_M", type=int, default=32)
    p.add_argument("--hnsw_efC", type=int, default=200)
    p.add_argument("--hnsw_efS", type=int, default=64)

    # 사전 필터
    p.add_argument("--prefilter", choices=["phash", "pdq", "both"], default="phash")
    p.add_argument("--phash_thresh", type=int, default=12)
    p.add_argument("--pdq_thresh", type=int, default=75)
    p.add_argument("--density_diff", type=float, default=0.20)

    # 유사도 임계값
    p.add_argument("--cnn_thresh", type=float, default=0.98)
    p.add_argument("--suspect_low", type=float, default=0.93)

    # 공백(빈칸) 감지
    p.add_argument("--blank_method", choices=["otsu", "sauvola"], default="sauvola")
    p.add_argument("--blank_thresh", type=float, default=0.02)

    # 재정렬 / OCR (선택)
    p.add_argument("--use_lpips", action="store_true")
    p.add_argument("--lpips_thresh", type=float, default=0.2)
    p.add_argument("--use_ocr", action="store_true")
    p.add_argument("--text_sim_thresh", type=float, default=0.85)

    # 정렬 (Alignment)
    p.add_argument("--use_alignment", action="store_true")

    # 임베딩
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--roi", type=float, nargs=4, default=[0.15, 0.15, 0.85, 0.85])

    # 자동 최적화
    p.add_argument("--no_auto_optimize", action="store_true", help="데이터 크기에 따른 자동 최적화 비활성화")

    # UI 옵션
    p.add_argument("--no_gui", action="store_true", help="폴더 선택 GUI 없이 동작(반드시 --input_dir 제공)")

    # 내부 인자 무시는 상위에서 처리하지 않음(단독 실행 파일)
    return p.parse_args()


def _choose_input_dir_with_gui() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    try:
        root = tk.Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        sel = filedialog.askdirectory(title="분석할 폴더 선택")
        try:
            root.destroy()
        except Exception:
            pass
        return sel or None
    except Exception:
        return None


def main():
    args = parse_args()

    # detect_pipeline import (패키지/스크립트 실행 모두 호환)
    try:
        try:
            from .detector_pipeline import detect_pipeline, DetectorConfig
        except Exception:
            from detector_pipeline import detect_pipeline, DetectorConfig
    except Exception as e:
        print(f"모듈 로드 실패: {e}")
        sys.exit(2)

    # 입력 폴더 결정
    input_dir = args.input_dir
    if not input_dir and not args.no_gui:
        print("📁 폴더를 선택하세요…")
        input_dir = _choose_input_dir_with_gui()
    if not input_dir:
        print("❌ 입력 폴더가 지정되지 않았습니다. --input_dir로 경로를 지정하거나 GUI를 사용하세요.")
        sys.exit(2)
    if not os.path.isdir(input_dir):
        print(f"❌ 입력 폴더가 존재하지 않습니다: {input_dir}")
        sys.exit(2)

    # 출력 폴더 준비는 detect_pipeline 내부에서 안전하게 수행함
    os.makedirs(args.output_dir, exist_ok=True)

    # 구성
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
        auto_optimize=not args.no_auto_optimize,
    )

    # 진행 상황(간단 콘솔)
    start_ts = time.time()

    def progress_printer(stage: str, pct: float = 0.0, msg: str = ""):
        try:
            pct_val = float(pct) if 0.0 <= pct <= 1.0 else 0.0
            if pct_val < 1.0:
                print(f"\r⏳ {stage:10s} {pct_val*100:5.1f}% - {msg}", end="", flush=True)
            else:
                print(f"\r✅ {stage:10s} 완료 - {msg}")
        except Exception:
            pass

    print("🔍 파이프라인 시작…")
    try:
        _pairs, _groups = detect_pipeline(
            input_dir,
            args.output_dir,
            config=cfg,
            recursive=args.recursive,
            progress_callback=progress_printer,
        )
    except Exception as e:
        print(f"\n❌ 파이프라인 실패: {e}")
        sys.exit(1)

    elapsed = time.time() - start_ts
    print(f"\n✅ 완료! 총 소요 {elapsed:.1f}s  결과 폴더: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    # Windows(PyInstaller) 멀티프로세싱 호환
    try:
        from multiprocessing import freeze_support, set_start_method
        freeze_support()
        try:
            set_start_method('spawn')
        except Exception:
            pass
    except Exception:
        pass
    main()
