import os
import argparse
import subprocess
import threading
import shutil
import stat
from typing import Optional, Tuple

# 가능한 한 일찍 백그라운드 스레드에서 tkinter를 예열하여
# 폴더 선택 대화상자가 요청될 때 더 빠르게 열리도록 합니다.
# 모듈 레벨에서는 가벼운 임포트만 유지하여 시작 시 차단을 방지합니다.
_tk_warmed: bool = False
_tk_mods: Optional[Tuple[object, object]] = None
def _warm_tk():
    global _tk_warmed, _tk_mods
    try:
        import tkinter as tk
        from tkinter import filedialog
    # main()가 즉시 사용할 수 있도록 모듈 참조를 유지합니다.
        _tk_mods = (tk, filedialog)
        _tk_warmed = True
    except Exception:
        _tk_warmed = False

# 모듈 import 시 즉시 예열을 시작합니다(데몬 스레드로 프로세스 종료를 방해하지 않습니다).
_tk_thread = threading.Thread(target=_warm_tk, daemon=True)
_tk_thread.start()

def parse_args():
    p = argparse.ArgumentParser(description="Answer Sheet QA — pipeline & dashboard (Handwriting-Optimized)")
    p.add_argument("--output_dir", default="output")

    # 백엔드
    p.add_argument("--embed_backend", choices=["resnet18", "dinov2"], default="dinov2")
    p.add_argument("--ann_backend", choices=["auto", "brute", "faiss", "hnsw"], default="auto")

    # ANN 파라미터
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--hnsw_M", type=int, default=32)
    p.add_argument("--hnsw_efC", type=int, default=200)
    p.add_argument("--hnsw_efS", type=int, default=64)

    # 사전 필터
    p.add_argument("--prefilter", choices=["phash", "pdq", "both"], default="phash")
    p.add_argument("--phash_thresh", type=int, default=10)
    p.add_argument("--pdq_thresh", type=int, default=80)
    p.add_argument("--density_diff", type=float, default=0.15)

    # 유사도 임계값
    p.add_argument("--cnn_thresh", type=float, default=0.99)
    p.add_argument("--suspect_low", type=float, default=0.95)

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

    # 대화상자를 위한 단기간의 루트를 생성하고 최상위로 표시되도록 합니다.
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

# 무거운 detector pipeline은 폴더 선택 대화상자를 표시한 이후에 지연 임포트합니다.
# 이렇게 하면 사용자가 느끼는 시작 지연이 줄어듭니다.
    print("🔍 탐지 실행…")
    try:
    # 패키지(python -m src.main)로 실행할 때는 상대 임포트가 작동합니다;
    # 스크립트(python src/main.py)로 실행할 때는 절대 임포트가 필요할 수 있습니다.
    # 먼저 상대 임포트를 시도하고 실패하면 절대 임포트로 대체합니다.
        try:
            from .detector_pipeline import detect_pipeline, DetectorConfig
        except Exception:
            from detector_pipeline import detect_pipeline, DetectorConfig
    except Exception as e:
        print(f"검사 도중 모듈을 불러오지 못했습니다: {e}")
        return

    # sel은 디렉터리 경로 문자열입니다. 비어 있으면 중단합니다.
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