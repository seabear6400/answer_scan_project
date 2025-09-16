import os
import argparse
import subprocess
import threading
import shutil
import stat
from typing import Optional, Tuple

# Pre-warm tkinter in a background thread as early as possible so the folder
# dialog opens faster when requested. Keep only lightweight imports at module
# import time to avoid blocking startup.
_tk_warmed: bool = False
_tk_mods: Optional[Tuple[object, object]] = None
def _warm_tk():
    global _tk_warmed, _tk_mods
    try:
        import tkinter as tk
        from tkinter import filedialog
        # Keep references to modules so main() can use them immediately.
        _tk_mods = (tk, filedialog)
        _tk_warmed = True
    except Exception:
        _tk_warmed = False

# Start warming immediately on import (daemon thread so it won't block exit).
_tk_thread = threading.Thread(target=_warm_tk, daemon=True)
_tk_thread.start()

def parse_args():
    p = argparse.ArgumentParser(description="Answer Sheet QA — pipeline & dashboard (Handwriting-Optimized)")
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
    # GUI로 폴더 선택: 사용자가 폴더를 선택하면 그 폴더를 분석합니다.
    try:
        if _tk_warmed and _tk_mods:
            tk, filedialog = _tk_mods
        else:
            import tkinter as tk
            from tkinter import filedialog

        # Create a short-lived root for the dialog and ensure it's on top.
        root = tk.Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        print("[*] 폴더 선택 대화상자를 엽니다 — 분석할 폴더를 선택하세요.")
        sel = filedialog.askdirectory(title="분석할 폴더 선택")
        try:
            root.destroy()
        except Exception:
            pass
    except Exception:
        print("파일 선택 UI를 초기화하지 못했습니다.")
        sel = ()
    # 안전한 초기화: output 하위의 기존 내용을 삭제(읽기전용 파일 처리)한 뒤 재생성합니다.
    def _handle_remove_readonly(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
        except Exception:
            pass
        try:
            func(path)
        except Exception:
            pass

    for sub in ["grouped", "ok", "blank_answers", "artifacts"]:
        out_sub = os.path.join(args.output_dir, sub)
        try:
            if os.path.exists(out_sub):
                shutil.rmtree(out_sub, onerror=_handle_remove_readonly)
        except Exception:
            # 삭제 실패 시 안전하게 넘어가고 기존 디렉터리를 덮어쓰지 않습니다.
            pass
        os.makedirs(out_sub, exist_ok=True)

    # Lazy-import the heavy detector pipeline only after we've shown the
    # folder-selection dialog. This reduces perceived startup latency.
    print("🔍 탐지 실행…")
    try:
        # When run as a package (python -m src.main) a relative import works;
        # when run as a script (python src/main.py) the absolute import may be
        # needed. Try relative first, then fall back to absolute.
        try:
            from .detector_pipeline import detect_pipeline, DetectorConfig
        except Exception:
            from detector_pipeline import detect_pipeline, DetectorConfig
    except Exception as e:
        print(f"검사 도중 모듈을 불러오지 못했습니다: {e}")
        return

    # sel is a directory path string. If empty, abort.
    if not sel:
        print("중단: 처리할 폴더가 선택되지 않았습니다.")
        return

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

    detect_pipeline(sel, args.output_dir, config=cfg)
    print("✅ 완료 → report.csv, report.parquet, images_summary.csv 생성")

    print("🌐 대시보드 실행…")
    # input_dir 인자도 함께 전달하여 사용자가 선택한 입력 폴더가 대시보드에서 인식되도록 함
    cmd = ["python", "-m", "streamlit", "run", "src/dashboard.py", "--",
           f"--output_dir={args.output_dir}"]
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