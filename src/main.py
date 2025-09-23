import os
import argparse
import subprocess
import threading
import shutil
import stat
from typing import Optional, Tuple
import time
import subprocess
import os

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
    # 파일 수집 재귀 옵션
    p.add_argument("--recursive", action="store_true", help="Recursively scan input_dir for images")
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
    # 사용자가 폴더를 선택한 직후의 타임스탬프를 기록합니다.
    # (요구사항: "폴더 선택 시점 → 대시보드 준비 완료"의 실제 경과를 측정)
    selection_ts = time.time() if sel else None
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

    # (예상 시간 계산 코드는 제거됨 — 요청에 따라 출력에서 제외합니다)

    # progress callback: 콘솔에 단계/퍼센트/메시지를 출력 (진행바 + ETA 포함)
    _progress_state = {"start": time.time(), "stages": {}}

    def _format_secs(s: Optional[float]) -> str:
        if s is None:
            return "--:--:--"
        s = int(round(s))
        h, r = divmod(s, 3600)
        m, s = divmod(r, 60)
        if h:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def progress_printer(stage: str, pct: float = 0.0, msg: str = ""):
        """
        개선된 ETA 계산:
        - msg에 'N/M' 형태가 있으면 우선으로 사용하여 pct를 계산합니다.
        - pct 기반으로 순간 속도(증분 pct / dt)를 EMA로 추적하여 ETA를 산출합니다.
        - 속도가 너무 작거나 불안정하면 elapsed*(1/pct - 1)로 폴백합니다.
        """
        try:
            import time as _time, re as _re
            now = _time.time()

            st = _progress_state["stages"].setdefault(
                stage,
                {
                    "first": now,
                    "last": now,
                    "last_pct": 0.0,
                    "msg": "",
                    "speed_ema": None,
                },
            )

            # msg에 "processed/total" 형식이 있으면 우선으로 pct 산출
            pct_val = None
            try:
                m = _re.search(r"(\d+)\s*/\s*(\d+)", str(msg or ""))
                if m:
                    p = int(m.group(1))
                    q = int(m.group(2))
                    if q > 0:
                        pct_val = max(0.0, min(1.0, float(p) / float(q)))
            except Exception:
                pct_val = None

            # 전달된 pct 인자를 사용해야 할 경우
            if pct_val is None:
                try:
                    pct_val = float(pct)
                    if not (0.0 <= pct_val <= 1.0):
                        pct_val = 0.0
                except Exception:
                    pct_val = 0.0

            # 최초 진입 시 first 설정
            if st["last_pct"] == 0.0 and pct_val > 0.0:
                st["first"] = st.get("first", now)
            dt = max(1e-6, now - st.get("last", now))
            # 순간 속도: 증분 pct / dt
            inst_speed = None
            try:
                inst_speed = (pct_val - st.get("last_pct", 0.0)) / dt
            except Exception:
                inst_speed = None

            # EMA 업데이트
            alpha = 0.25
            prev_ema = st.get("speed_ema", None)
            if inst_speed is not None:
                if prev_ema is None:
                    speed_ema = inst_speed
                else:
                    speed_ema = alpha * inst_speed + (1.0 - alpha) * prev_ema
            else:
                speed_ema = prev_ema

            # 안전 장치: 음수 또는 너무 작은 속도는 무시
            min_speed = 1e-5
            if speed_ema is None or speed_ema < min_speed:
                speed_use = None
            else:
                speed_use = speed_ema

            # ETA 계산: 남은 pct / speed
            eta = None
            elapsed = now - st.get("first", now)
            try:
                if speed_use is not None and pct_val < 0.999999:
                    eta = (1.0 - pct_val) / speed_use
                else:
                    # 폴백: elapsed*(1/pct - 1)
                    if pct_val > 1e-6:
                        eta = elapsed * (1.0 / pct_val - 1.0)
                    else:
                        eta = None
            except Exception:
                eta = None

            # 상태 갱신
            st["last"] = now
            st["last_pct"] = pct_val
            st["msg"] = str(msg)
            st["speed_ema"] = speed_ema

            # 출력(콘솔 간단 로그 유지)
            try:
                eta_s = _format_secs(eta)
                elapsed_s = _format_secs(elapsed)
                print(f"[진행] {stage:12s} {pct_val*100:5.1f}%  ETA:{eta_s}  경과:{elapsed_s}  {msg}")
            except Exception:
                print(f"[진행] {stage} | {pct_val} | {msg}")

        except Exception:
            try:
                print(f"[진행] {stage} | {pct} | {msg}")
            except Exception:
                pass

    # 사전 검사: 선택된 폴더에 지원 이미지 확장자가 있는지 확인합니다.
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    detected = []
    try:
        if args.recursive:
            for root, _dirs, files in os.walk(sel):
                for fn in files:
                    if fn.lower().endswith(exts):
                        detected.append(os.path.join(root, fn))
        else:
            for fn in sorted(os.listdir(sel)):
                if fn.lower().endswith(exts):
                    detected.append(os.path.join(sel, fn))
    except Exception as e:
        print(f"폴더 검사 중 예외: {e}")

    if not detected:
        print("오류: 선택한 폴더에 지원 이미지 파일이 없습니다.")
        print(f"선택 폴더: {sel}")
        try:
            entries = sorted(os.listdir(sel))[:20]
            print(f"폴더 상위 항목(최대 20개): {entries}")
        except Exception:
            pass
        print("확인 및 해결:")
        print(" - 이미지 확장자가 jpg/png/tif 등인지 확인하세요.")
        print(" - OneDrive가 파일을 '온라인 전용'으로 만들었는지 확인하세요(파일을 로컬로 동기화).")
        print(" - 하위 폴더의 이미지를 포함하려면 --recursive 옵션을 사용하세요.")
        return

    # 파이프라인을 백그라운드 스레드에서 실행하고,
    # 메인(UI) 스레드에서는 카톡 스타일의 토스트 창을 띄워 진행을 보여줍니다.
    try:
        import tkinter as tk
        from tkinter import ttk

        class ToastToast:
            def __init__(self, shared_state, worker_thread, width=480, height=140, stay_time=None):
                self.shared = shared_state
                self.worker = worker_thread
                self.width = width
                self.height = height
                self.stay_time = stay_time
                self.root = tk.Tk()
                # 창 꾸밈: 테두리 없이 항상 위, 투명도 약간, 포커스 강제 X
                try:
                    self.root.overrideredirect(True)
                except Exception:
                    pass
                self.root.attributes("-topmost", True)
                try:
                    self.root.attributes("-alpha", 0.95)
                except Exception:
                    pass

                # 화면 중앙에 배치
                sw = self.root.winfo_screenwidth()
                sh = self.root.winfo_screenheight()
                x = int((sw - self.width) / 2)
                y = int((sh - self.height) / 2)
                try:
                    self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")
                except Exception:
                    pass

                # 스타일/위젯
                frm = tk.Frame(self.root, bg="#ffffff", bd=1, relief="solid")
                frm.pack(fill="both", expand=True)
                self.title_var = tk.StringVar(value="진행 대기...")
                self.msg_var = tk.StringVar(value="")
                self.eta_var = tk.StringVar(value="ETA: --:--")
                tk.Label(frm, textvariable=self.title_var, font=("Segoe UI", 10, "bold"), bg="#ffffff").pack(anchor="w", padx=10, pady=(8,0))
                self.pb = ttk.Progressbar(frm, orient="horizontal", length=self.width-24, mode="determinate")
                self.pb.pack(padx=10, pady=(6,2))
                tk.Label(frm, textvariable=self.msg_var, font=("Segoe UI", 9), bg="#ffffff").pack(anchor="w", padx=10)
                tk.Label(frm, textvariable=self.eta_var, font=("Segoe UI", 9), bg="#ffffff").pack(anchor="w", padx=10, pady=(4,8))

                # 마우스 클릭 시 최소화/닫기 처리
                frm.bind("<Button-1>", lambda e: self._on_click())
                # 창 드래그(이동) 지원: 마우스 드래그로 위치 변경
                def start_move(event):
                    try:
                        self._drag_start_x = event.x
                        self._drag_start_y = event.y
                    except Exception:
                        self._drag_start_x = None
                        self._drag_start_y = None

                def do_move(event):
                    try:
                        if getattr(self, '_drag_start_x', None) is None:
                            return
                        dx = event.x - self._drag_start_x
                        dy = event.y - self._drag_start_y
                        geom = self.root.geometry().split('+')
                        if len(geom) >= 3:
                            cur_x = int(geom[1])
                            cur_y = int(geom[2])
                            new_x = cur_x + dx
                            new_y = cur_y + dy
                            self.root.geometry(f"{self.width}x{self.height}+{new_x}+{new_y}")
                    except Exception:
                        pass

                frm.bind('<ButtonPress-1>', start_move)
                frm.bind('<B1-Motion>', do_move)
                self.root.protocol("WM_DELETE_WINDOW", self._on_close)
                self._start_time = time.time()
                self._update_loop()

            def _format_secs(self, s: Optional[float]) -> str:
                if s is None:
                    return "--:--:--"
                s = int(round(s))
                h, r = divmod(s, 3600)
                m, s = divmod(r, 60)
                if h:
                    return f"{h:02d}:{m:02d}:{s:02d}"
                return f"{m:02d}:{s:02d}"

            def _on_click(self):
                # 클릭하면 윈도우를 닫지 않고 아이콘화(최소화) — 사용자가 보고 싶을 때 다시 표시 가능
                try:
                    self.root.iconify()
                except Exception:
                    pass

            def _on_close(self):
                # 파이프라인이 끝나기 전에는 닫지 않음(강제로 닫고 싶으면 아이콘화)
                if self.worker.is_alive():
                    try:
                        self.root.iconify()
                    except Exception:
                        pass
                else:
                    try:
                        self.root.destroy()
                    except Exception:
                        pass

            def _update_loop(self):
                try:
                    stages = self.shared.get("stages", {})
                    if stages:
                        latest = max(stages.items(), key=lambda kv: kv[1].get("last", 0))[0]
                        info = stages.get(latest, {})
                        pct = info.get("last_pct", 0.0)
                        msg = info.get("msg", "")
                        first = info.get("first", time.time())
                        elapsed = time.time() - first
                        eta = None
                        if pct > 0:
                            try:
                                eta = elapsed * (1.0 / pct - 1.0)
                            except Exception:
                                eta = None
                        self.title_var.set(f"단계: {latest}")
                        self.msg_var.set(str(msg) or "")
                        self.eta_var.set(f"경과: {self._format_secs(elapsed)}  ETA: {self._format_secs(eta)}")
                        try:
                            self.pb['value'] = max(0.0, min(100.0, pct * 100.0))
                        except Exception:
                            pass
                    # 파이프라인 완료 시 자동 닫기(짧은 딜레이 후)
                    if not self.worker.is_alive():
                        try:
                            # 완료 표시 후 0.8s 뒤 닫기
                            self.root.after(800, self.root.destroy)
                            return
                        except Exception:
                            pass
                except Exception:
                    pass
                # 200ms마다 갱신
                self.root.after(200, self._update_loop)

            def run(self):
                try:
                    self.root.mainloop()
                except Exception:
                    pass

        # 파이프라인 스레드 시작
        pipeline_thread = threading.Thread(
            target=lambda: detect_pipeline(sel, args.output_dir, config=cfg, recursive=args.recursive, progress_callback=progress_printer),
            daemon=True
        )
        pipeline_thread.start()

        # 토스트 창을 메인 스레드에서 실행 (blocking until closed)
        try:
            toast = ToastToast(_progress_state, pipeline_thread)
            toast.run()
        except Exception as e:
            print(f"토스트 창 실행 실패: {e}")
            # 실패하면 블록킹 방식으로 대체 실행
            pipeline_thread.join()
    except Exception:
        # tkinter가 없거나 실패 시 기존 동기 호출로 폴백
        detect_pipeline(sel, args.output_dir, config=cfg, recursive=args.recursive, progress_callback=progress_printer)
    print("✅ 완료 → report.csv, report.parquet, images_summary.csv 생성")

    print("🌐 대시보드 실행…")
    # input_dir 인자도 함께 전달하여 사용자가 선택한 입력 폴더가 대시보드에서 인식되도록 함
    cmd = ["python", "-m", "streamlit", "run", "src/dashboard.py", "--",
           f"--output_dir={args.output_dir}"]
    # 기준 시점: 사용자가 폴더를 선택한 시점을 우선 사용, 없으면 지금부터 측정
    start_to_dashboard = selection_ts or time.time()

    def _wait_streamlit_ready_and_report(proc, timeout: int = 90):
        ready_patterns = ("Local URL:", "Network URL:", "You can now view your Streamlit app", "Running on")
        t0 = time.time()
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                print(f"[streamlit] {line}")
                if any(p in line for p in ready_patterns):
                    elapsed = time.time() - start_to_dashboard
                    print(f"🎉 대시보드 준비 완료 (선택→표시): {elapsed:.2f}s")
                    return elapsed
                if time.time() - t0 > timeout:
                    print(f"⚠️ Streamlit 준비 대기 타임아웃({timeout}s)")
                    return None
        except Exception as e:
            print(f"Streamlit 모니터링 예외: {e}")
            return None

    try:
        if args.detach and os.name == 'nt':
            # 윈도우에서 새 창으로 띄우는 경우에는 stdout 캡처가 어렵고 start 명령으로 즉시 반환됩니다.
            subprocess.Popen(["cmd", "/c", "start"] + cmd)
            print(f"대시보드 프로세스 시작 (detach). 경과(근사): {time.time() - start_to_dashboard:.2f}s")
        else:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
            _wait_streamlit_ready_and_report(proc, timeout=90)
    except KeyboardInterrupt:
        print("중단: 사용자가 실행을 취소했습니다.")
    except Exception as e:
        print(f"대시보드 실행 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()