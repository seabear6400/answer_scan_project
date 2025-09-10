"""
간단한 GUI 래퍼: 입력/출력 디렉토리 선택, 파이프라인 실행, Streamlit 대시보드 실행

사용법:
    python src/gui.py

이 스크립트는 외부 의존성을 추가하지 않고 표준 라이브러리(Tkinter, threading, subprocess)를 사용합니다.
"""
import os
import threading
import subprocess
import sys
import traceback
from tkinter import Tk, Label, Entry, Button, Text, END, filedialog, StringVar, IntVar, Checkbutton

try:
    # 로컬 모듈 임포트
    from detector_pipeline import detect_pipeline, DetectorConfig
except Exception:
    detect_pipeline = None
    DetectorConfig = None


def choose_dir(var: StringVar):
    d = filedialog.askdirectory()
    if d:
        var.set(d)


def append_log(txt_widget: Text, s: str):
    txt_widget.insert(END, s + "\n")
    txt_widget.see(END)


def run_detection(input_dir: str, output_dir: str, opts: dict, log_widget: Text, enable_dashboard_btn):
    try:
        if detect_pipeline is None or DetectorConfig is None:
            append_log(log_widget, "오류: detector_pipeline 모듈을 불러올 수 없습니다. src 폴더 경로에서 실행 중인지 확인하세요.")
            return

        os.makedirs(output_dir, exist_ok=True)
        append_log(log_widget, f"🔍 파이프라인 시작: input={input_dir} output={output_dir}")

        cfg = DetectorConfig(
            embed_backend=opts.get('embed_backend', 'dinov2'),
            ann_backend=opts.get('ann_backend', 'auto'),
            k=int(opts.get('k', 20)),
            hnsw_M=int(opts.get('hnsw_M', 32)),
            hnsw_efC=int(opts.get('hnsw_efC', 200)),
            hnsw_efS=int(opts.get('hnsw_efS', 64)),
            prefilter=opts.get('prefilter', 'phash'),
            phash_thresh=int(opts.get('phash_thresh', 10)),
            pdq_thresh=int(opts.get('pdq_thresh', 80)),
            density_diff_thresh=float(opts.get('density_diff', 0.15)),
            cnn_thresh=float(opts.get('cnn_thresh', 0.99)),
            suspect_low=float(opts.get('suspect_low', 0.95)),
            blank_method=opts.get('blank_method', 'sauvola'),
            blank_density_thresh=float(opts.get('blank_thresh', 0.02)),
            use_lpips=bool(opts.get('use_lpips', False)),
            lpips_thresh=float(opts.get('lpips_thresh', 0.2)),
            use_ocr=bool(opts.get('use_ocr', False)),
            text_sim_thresh=float(opts.get('text_sim_thresh', 0.85)),
            use_alignment=bool(opts.get('use_alignment', False)),
            batch_size=int(opts.get('batch', 64)),
            num_workers=int(opts.get('num_workers', 0)),
            roi_ratio=tuple(opts.get('roi', (0.15, 0.15, 0.85, 0.85))),
        )

        # 로그 콜백을 간단히 print로 대체 (detect_pipeline 내부가 print를 사용하면 텍스트에 표시되지 않음)
        # 간단한 통합: monitor 파일 변경이 있을 경우 텍스트에 쓰도록 후처리
        detect_pipeline(input_dir, output_dir, config=cfg)

        append_log(log_widget, "✅ 파이프라인 완료 — output 디렉토리를 확인하세요.")
        enable_dashboard_btn(True)
    except Exception as e:
        append_log(log_widget, "파이프라인 실행 중 예외 발생:")
        append_log(log_widget, ''.join(traceback.format_exception_only(type(e), e)))
        enable_dashboard_btn(False)


def launch_dashboard(output_dir: str, log_widget: Text):
    try:
        append_log(log_widget, "🌐 Streamlit 대시보드 실행 중...")
        cmd = [sys.executable, "-m", "streamlit", "run", "src/dashboard.py", "--", f"--output_dir={output_dir}"]
        # Windows에서 새 창으로 띄우려면 cmd start 사용
        if os.name == 'nt':
            subprocess.Popen(["cmd", "/c", "start"] + cmd)
        else:
            subprocess.Popen(cmd)
        append_log(log_widget, "대시보드가 새 창에서 실행되었습니다.")
    except Exception as e:
        append_log(log_widget, f"대시보드 실행 실패: {e}")


def main():
    root = Tk()
    root.title("Answer Scan — GUI")
    root.geometry("760x520")

    Label(root, text="입력 폴더:").grid(row=0, column=0, sticky='w', padx=6, pady=6)
    in_var = StringVar(value=os.path.abspath("input_images"))
    Entry(root, textvariable=in_var, width=60).grid(row=0, column=1, padx=6)
    Button(root, text="...", command=lambda: choose_dir(in_var)).grid(row=0, column=2, padx=6)

    Label(root, text="출력 폴더:").grid(row=1, column=0, sticky='w', padx=6, pady=6)
    out_var = StringVar(value=os.path.abspath("output"))
    Entry(root, textvariable=out_var, width=60).grid(row=1, column=1, padx=6)
    Button(root, text="...", command=lambda: choose_dir(out_var)).grid(row=1, column=2, padx=6)

    # 최소 옵션: 실행 버튼
    log = Text(root, height=18, width=90)
    log.grid(row=3, column=0, columnspan=3, padx=6, pady=6)

    def set_dashboard_enabled(enabled: bool):
        if enabled:
            btn_dash.config(state='normal')
        else:
            btn_dash.config(state='disabled')

    def on_run():
        input_dir = in_var.get().strip() or "input_images"
        output_dir = out_var.get().strip() or "output"
        # 단순화: 최소 옵션만 전달. 필요하면 위젯 추가로 확장 가능
        opts = {
            'embed_backend': 'dinov2', 'ann_backend': 'auto', 'k': 20,
            'hnsw_M': 32, 'hnsw_efC': 200, 'hnsw_efS': 64,
            'prefilter': 'phash', 'phash_thresh': 10, 'pdq_thresh': 80,
            'density_diff': 0.15, 'cnn_thresh': 0.99, 'suspect_low': 0.95,
            'blank_method': 'sauvola', 'blank_thresh': 0.02,
            'use_lpips': False, 'lpips_thresh': 0.2, 'use_ocr': False,
            'text_sim_thresh': 0.85, 'use_alignment': False,
            'batch': 64, 'num_workers': 0, 'roi': (0.15,0.15,0.85,0.85)
        }

        append_log(log, "스레드에서 파이프라인을 실행합니다...")
        set_dashboard_enabled(False)
        t = threading.Thread(target=run_detection, args=(input_dir, output_dir, opts, log, set_dashboard_enabled), daemon=True)
        t.start()

    def on_launch():
        out = out_var.get().strip() or "output"
        launch_dashboard(out, log)

    btn_run = Button(root, text="▶ 파이프라인 실행", command=on_run, width=20)
    btn_run.grid(row=2, column=1, sticky='w', padx=6)

    btn_dash = Button(root, text="🌐 대시보드 열기", command=on_launch, width=20, state='disabled')
    btn_dash.grid(row=2, column=1, sticky='e', padx=6)

    append_log(log, "Ready. '파이프라인 실행'을 눌러 시작하세요.")

    root.mainloop()


if __name__ == '__main__':
    main()
