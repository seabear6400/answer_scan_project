import argparse
import csv
import logging
import os
import shutil
import stat
import subprocess
import threading
import time
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import tempfile

# OpenCV 로깅 레벨 설정 (경고 메시지 숨김)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'



def _format_duration(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    try:
        total = int(round(max(0.0, seconds)))
    except Exception:
        return None
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours}시간")
    if minutes or hours:
        parts.append(f"{minutes}분")
    parts.append(f"{secs}초")
    return " ".join(parts)


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _count_csv_rows(csv_path: Path) -> Optional[int]:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            with csv_path.open("r", encoding=enc, newline="") as fh:
                reader = csv.reader(fh)
                next(reader, None)
                return sum(1 for _ in reader)
        except FileNotFoundError:
            return None
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    return None


def _count_image_files(root: Path) -> Optional[int]:
    if not root.exists():
        return None
    total = 0
    try:
        for current_root, _dirs, files in os.walk(root):
            for fname in files:
                if fname.lower().endswith(IMAGE_EXTS):
                    total += 1
    except Exception:
        return None
    return total


def _total_processed_items(result_paths) -> Optional[int]:
    if not result_paths:
        return None
    total = 0
    any_found = False
    for raw in result_paths:
        if not raw:
            continue
        summary_path = Path(raw) / "images_summary.csv"
        if summary_path.exists():
            count = _count_csv_rows(summary_path)
            if count is not None:
                total += count
                any_found = True
                continue
        fallback = _count_image_files(Path(raw))
        if fallback is not None:
            total += fallback
            any_found = True
    return total if any_found else None


def _print_run_summary(summary: Optional[dict], duration: Optional[float]) -> None:
    readable_duration = _format_duration(duration)
    if not summary:
        if readable_duration:
            print(f"⏱️ 소요 시간: {readable_duration}")
        return

    mode = summary.get("mode")
    runs = int(summary.get("runs", 0))
    result_paths = summary.get("result_paths") or []
    total_scopes = len(summary.get("scopes") or [])
    dataset_label = "응시 데이터" if mode == "scoped" else "폴더"
    dataset_count = len(result_paths) or runs or total_scopes or (1 if mode == "single" else 0)

    if dataset_count:
        print(f"📦 총 {dataset_count}개의 {dataset_label} 분석을 완료했습니다.")
    else:
        print("📦 분석이 완료되었습니다.")
    processed_total = _total_processed_items(result_paths)
    if processed_total is not None:
        print(f"📊 처리 데이터 수: {processed_total}개")
    else:
        print(f"📊 처리 데이터 수: {dataset_count}개")

    if readable_duration:
        print(f"⏱️ 소요 시간: {readable_duration}")

    issues = summary.get("issues", []) or []
    if issues:
        print("⚠️ 추가 확인이 필요한 항목:")
        for issue in issues:
            print(f" - {issue}")


def _result_dir_has_payload(path: Path) -> bool:
    markers = ("report.parquet", "report.csv", "images_summary.csv")
    for marker in markers:
        try:
            if (path / marker).exists():
                return True
        except Exception:
            continue
    try:
        grouped = path / "grouped"
        if grouped.exists():
            for _child in grouped.iterdir():
                return True
    except Exception:
        pass
    return False


def _ensure_result_zips(result_paths: Optional[List[str]]) -> List[str]:
    """주어진 결과 폴더 목록에 대해 ZIP 생성 시도를 수행하고, 성공한 ZIP 경로 목록을 반환합니다."""
    if not result_paths:
        return []
    try:
        # Pillow 등 다른 모듈과 유사하게 상대/절대 임포트 모두 대응합니다.
        try:
            from .detector_pipeline import create_result_zip_for_dir  # type: ignore
        except Exception:
            from detector_pipeline import create_result_zip_for_dir  # type: ignore
    except Exception:
        return []

    created: List[str] = []
    for raw in result_paths:
        if not raw:
            continue
        try:
            target = Path(raw).resolve()
        except Exception:
            target = Path(raw)
        if not target.exists() or not target.is_dir():
            continue
        try:
            zip_path = create_result_zip_for_dir(str(target))
            if zip_path:
                try:
                    created.append(str(Path(zip_path).resolve()))
                except Exception:
                    created.append(str(zip_path))
        except Exception:
            continue
    return created


def parse_args():

    # PyInstaller로 빌드한 실행 파일이 멀티프로세싱을 사용할 때
    # '--multiprocessing-fork ...' 같은 내부 인자를 전달하는데,
    # 이를 무시하도록 parse_known_args를 사용합니다.
    p = argparse.ArgumentParser(description="Answer Sheet QA — pipeline & dashboard (Handwriting-Optimized)")
    p.add_argument("--output_dir", default=None)

    # ANN 파라미터
    p.add_argument("--k", type=int, default=12)

    # 사전 필터
    p.add_argument("--prefilter", choices=["phash", "none"], default="phash")
    p.add_argument("--phash_thresh", type=int, default=10)
    p.add_argument("--density_diff", type=float, default=0.15)

    # 유사도 임계값
    p.add_argument("--cnn_thresh", type=float, default=0.99)
    p.add_argument("--suspect_low", type=float, default=0.95)

    # 공백(빈칸) 감지
    p.add_argument("--blank_method", choices=["otsu", "sauvola"], default="sauvola")
    p.add_argument("--blank_thresh", type=float, default=0.02)

    # 임베딩
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--roi", type=float, nargs=4, default=[0.15, 0.15, 0.85, 0.85])
    
    # 자동 최적화
    p.add_argument("--no_auto_optimize", action="store_true", help="데이터 크기에 따른 자동 최적화 비활성화")
    
    p.add_argument("--detach", action="store_true", help="윈도우에서 Streamlit을 새 창으로 분리 실행합니다 (비차단).")
    # 파일 수집 재귀 옵션
    p.add_argument("--recursive", action="store_true", help="선택한 폴더에서 하위 폴더까지 재귀적으로 이미지를 검색합니다")
    # 알 수 없는 내부 인자(예: --multiprocessing-fork ...)는 무시
    args, _unknown = p.parse_known_args()
    return args

def main():
    args = parse_args()
    # GUI로 폴더 선택: 사용자가 폴더를 선택하면 그 폴더를 분석합니다.
    try:
        import tkinter as tk
        from tkinter import filedialog

        # 대화상자를 위한 단기간의 루트를 생성하고 최상위로 표시되도록 합니다.
        root = tk.Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        print("📁 폴더를 선택하세요...")
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

    # sel은 디렉터리 경로 문자열입니다. 비어 있으면 중단합니다.
    if not sel:
        print("중단: 처리할 폴더가 선택되지 않았습니다.")
        return

    # 무거운 detector pipeline은 폴더 선택 대화상자를 표시한 이후에 지연 임포트합니다.
    print("🔍 탐지 시작...")
    summary: Optional[dict] = None
    default_result_path: Optional[str] = None

    try:
        # 패키지(python -m src.main)로 실행할 때는 상대 임포트가 작동합니다;
        # 스크립트(python src/main.py)로 실행할 때는 절대 임포트가 필요할 수 있습니다.
        # 먼저 상대 임포트를 시도하고 실패하면 절대 임포트로 대체합니다.
        try:
            from .detector_pipeline import DetectorConfig
            from .scope_runner import discover_exam_scopes, run_scoped_pipeline
        except Exception:
            from detector_pipeline import DetectorConfig
            from scope_runner import discover_exam_scopes, run_scoped_pipeline
    except Exception as e:
        print(f"검사 도중 모듈을 불러오지 못했습니다: {e}")
        return

    try:
        pre_scopes, _pre_issues = discover_exam_scopes(sel)
    except Exception:
        pre_scopes, _pre_issues = [], []

    multi_scope_mode = len(pre_scopes) > 0
    sel_path = Path(sel).resolve()
    def _compute_effective_output_dir(sel_path: Path, args, multi_scope_mode: bool) -> str:
        """안전하게 결과 폴더 경로를 결정합니다.

        규칙 요약:
        - multi_scope_mode인 경우 선택한 폴더를 결과 베이스로 사용합니다.
        - --output_dir가 지정되면 그 값을 우선 사용합니다.
        - 사용자가 이미 `_결과` 폴더를 선택한 경우 그대로 사용합니다.
        - 사용자가 `_결과` 내부(하위) 항목을 선택한 경우 상위 `_결과`를 재사용하여
          중첩된 `_결과` 폴더 생성을 방지합니다.
        - 그 외에는 선택 폴더와 같은 레벨에 `{selname}_결과` 이름의 형제 폴더를 사용합니다.
        """
        if multi_scope_mode:
            return str(sel_path)
        if args.output_dir:
            return str(Path(args.output_dir).expanduser().resolve())
        # 이미 결과 폴더를 선택한 경우 그대로 사용
        if sel_path.name.endswith("_결과"):
            return str(sel_path)
        # 선택 폴더가 결과 폴더 내부에 있는 경우 상위 *_결과 폴더를 재사용
        try:
            parent = sel_path.parent
            if parent.name.endswith("_결과"):
                return str(parent)
        except Exception:
            pass
        # 기본: 같은 레벨에 새 *_결과 폴더 생성 (단, sel_path와 동일 경로가 되지 않도록 보호)
        try:
            candidate = sel_path.with_name(f"{sel_path.name}_결과")
            if candidate == sel_path:
                return str(sel_path)
            return str(candidate)
        except Exception:
            return str(sel_path)

    effective_output_dir = _compute_effective_output_dir(sel_path, args, multi_scope_mode)

    if not multi_scope_mode:
        def _handle_remove_readonly(func, path, exc_info):
            try:
                os.chmod(path, stat.S_IWRITE)
            except Exception:
                pass
            try:
                func(path)
            except Exception:
                pass

        artifacts_dir = os.path.join(effective_output_dir, "artifacts")
        try:
            if os.path.exists(artifacts_dir):
                shutil.rmtree(artifacts_dir, onerror=_handle_remove_readonly)
        except Exception:
            pass
        os.makedirs(artifacts_dir, exist_ok=True)

    cfg = DetectorConfig(
        k=args.k,
        prefilter=args.prefilter,
        phash_thresh=args.phash_thresh,
        density_diff_thresh=args.density_diff,
        cnn_thresh=args.cnn_thresh,
        suspect_low=args.suspect_low,
        blank_method=args.blank_method,
        blank_density_thresh=args.blank_thresh,
        batch_size=args.batch,
        num_workers=args.num_workers,
        roi_ratio=tuple(args.roi),
        auto_optimize=not args.no_auto_optimize,
    )

    _progress_state = {"start": time.time(), "stages": {}}

    # 진행 상황 표시 전략 선택
    def progress_printer_silent(stage: str, pct: float = 0.0, msg: str = ""):
        """토스트 창용 - 콘솔 출력은 숨기지만 상태는 업데이트"""
        try:
            import time as _time
            now = _time.time()
            
            # 상태 관리 - 토스트 창이 읽을 수 있도록 업데이트
            st = _progress_state["stages"].setdefault(stage, {"first": now})
            st["last"] = now
            st["last_pct"] = float(pct) if 0.0 <= pct <= 1.0 else 0.0
            st["msg"] = str(msg)
            
        except Exception:
            pass
        
    def progress_printer_console(stage: str, pct: float = 0.0, msg: str = ""):
        """토스트 창 실패 시 콘솔 출력용"""
        try:
            import time as _time
            now = _time.time()
            
            # 간단한 상태 관리
            st = _progress_state["stages"].setdefault(stage, {"first": now})
            
            # 전달된 pct 사용
            pct_val = float(pct) if 0.0 <= pct <= 1.0 else 0.0
            
            # ETA 계산
            if pct_val > 0.01:
                elapsed = now - st["first"]
                eta = elapsed * (1.0 / pct_val - 1.0)
                eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"
            else:
                eta_str = "--:--"
            
            # 간단한 출력
            if pct_val < 1.0:
                print(f"\r⏳ {stage}: {pct_val*100:5.1f}% (ETA: {eta_str}) - {msg}", end="", flush=True)
            else:
                print(f"\r✅ {stage}: 완료 - {msg}")
                
        except Exception:
            print(f"⏳ {stage}: {msg}")

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

    if not detected and pre_scopes:
        print(f"선택된 폴더에서 {len(pre_scopes)}개의 응시 데이터 폴더를 발견했습니다. 해당 폴더의 이미지를 분석합니다.")

    if not detected and not pre_scopes:
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
    # 메인(UI) 스레드에서는 토스트 창을 띄워 진행을 보여줍니다.
    start_to_dashboard = selection_ts or time.time()

    _run_summary: dict = {}
    first_scope_event = threading.Event()
    first_scope_info: dict = {}
    dashboard_lock = threading.Lock()
    dashboard_state = {"started": False, "proc": None}

    def _wait_streamlit_ready_and_report(proc, timeout: int = 90):
        ready_patterns = ("Local URL:", "Network URL:", "You can now view your Streamlit app", "Running on")
        t0 = time.time()
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if any(p in line for p in ready_patterns):
                    elapsed = time.time() - start_to_dashboard
                    print(f"\n✅ 대시보드 준비 완료! ({elapsed:.1f}초)")
                    return elapsed
                if time.time() - t0 > timeout:
                    print(f"⚠️ 대시보드 시작 대기 시간 초과 ({timeout}s)")
                    return None
        except Exception as e:
            print(f"대시보드 모니터링 오류: {e}")
            return None

    def _resolve_top_level_base(raw_base: Optional[str]) -> str:
        try:
            selection_root = sel_path.resolve()
        except Exception:
            selection_root = sel_path
        if not raw_base:
            return str(selection_root)
        try:
            base_path = Path(raw_base).resolve()
        except Exception:
            base_path = Path(raw_base)
        try:
            base_path.relative_to(selection_root)
            return str(selection_root)
        except Exception:
            pass
        while base_path.name.endswith("_결과") and base_path.parent != base_path:
            base_path = base_path.parent
            try:
                base_path.relative_to(selection_root)
                return str(selection_root)
            except Exception:
                continue
        return str(base_path)

    def _maybe_launch_dashboard(
        trigger: str,
        output_dir: Optional[str],
        base_dir: Optional[str],
        default_result: Optional[str],
    ) -> None:
        if not output_dir:
            return

        try:
            resolved_output = str(Path(output_dir).resolve())
        except Exception:
            resolved_output = str(output_dir)

        resolved_base = base_dir
        if not resolved_base:
            if multi_scope_mode:
                resolved_base = str(sel_path)
            else:
                base_candidate = Path(effective_output_dir)
                parent_candidate = base_candidate.parent if base_candidate.parent != base_candidate else base_candidate
                resolved_base = str(parent_candidate)
        try:
            resolved_base = str(Path(resolved_base).resolve())
        except Exception:
            resolved_base = str(resolved_base)

        resolved_base = _resolve_top_level_base(resolved_base)

        resolved_default = default_result or resolved_output

        if getattr(sys, "frozen", False):
            # frozen 상태에서는 _MEIPASS(임시 추출 폴더)와 exe 옆 폴더를 후보로 검사합니다.
            # 문제 사례: AV 또는 OS 권한 제한으로 _MEIPASS 내부 파일 접근이 거부될 수 있음.
            # 따라서 우선 exe 옆의 dashboard.py를 시도하고, _MEIPASS는 후순위로 처리합니다.
            candidates = []
            try:
                exe_dir = Path(sys.executable).resolve().parent
            except Exception:
                exe_dir = Path(sys.executable).parent
            # 1) 우선 실행파일 옆에 있는 dashboard.py 우선 사용
            candidates.append(exe_dir / "dashboard.py")
            try:
                meipass = getattr(sys, "_MEIPASS", None)
            except Exception:
                meipass = None
            if meipass:
                # 2) PyInstaller가 임시로 추출한 경로(후보)
                candidates.append(Path(meipass) / "dashboard.py")

            dashboard_py_path = None
            for cand in candidates:
                try:
                    # exists() 호출 자체가 PermissionError를 던질 수 있으므로 안전하게 처리
                    if not cand.exists():
                        continue
                    # 실제로 읽을 수 있는지 확인 (권한 검사)
                    try:
                        with cand.open("rb"):
                            pass
                        dashboard_py_path = cand
                        break
                    except PermissionError:
                        print(f"⚠️ 권한 거부(읽기 불가): {cand} — 다음 후보 검사")
                        continue
                    except Exception:
                        # 읽기 불가면 다음 후보로
                        continue
                except PermissionError:
                    print(f"⚠️ 권한 거부(존재 검사 중): {cand} — 다음 후보 검사")
                    continue
                except Exception:
                    continue
            # 어떤 후보도 유효하지 않으면 exe 옆 후보(첫번째)를 기본값으로 두고 아래에서 추가 처리
            if dashboard_py_path is None:
                dashboard_py_path = candidates[0]

            # frozen 상태라면 권한/잠금 문제를 완전히 회피하기 위해
            # 원본 dashboard.py를 임시 폴더로 복사한 뒤 그 복사본을 실행하도록 강제합니다.
            try:
                # tmp에 복사하여 안전하게 실행
                tmp_dir = Path(tempfile.mkdtemp(prefix="answer_scan_dash_"))
                tmp_dashboard = tmp_dir / "dashboard.py"
                try:
                    shutil.copy2(str(dashboard_py_path), str(tmp_dashboard))
                    dashboard_py_path = tmp_dashboard
                    print(f"ℹ️ 대시보드 스크립트를 임시 폴더로 복사하여 실행합니다: {dashboard_py_path}")
                except Exception as copy_exc:
                    print(f"⚠️ 대시보드 임시 복사 실패: {copy_exc} — 원본 경로 사용 시도")
            except Exception:
                # tmp 디렉터리 생성/복사 실패 시 무시하고 원본 경로 사용
                pass
        else:
            dashboard_py_path = Path(__file__).with_name("dashboard.py")

        dashboard_py = str(dashboard_py_path)
        # 파일이 존재하지만 권한 문제로 읽을 수 없으면 임시 파일로 복사하여 사용을 시도합니다.
        db_path_obj = Path(dashboard_py)
        if not db_path_obj.exists():
            print(f"⚠️ 대시보드 스크립트를 찾을 수 없습니다: {dashboard_py}")
            print("   PyInstaller 빌드 시 dashboard.py를 데이터 파일로 포함했는지 확인하세요.")
            return
        try:
            with db_path_obj.open("rb") as _f:
                pass
        except PermissionError:
            # 읽기 권한이 없으면, 안전하게 임시 파일로 복사하여 대시보드를 실행하도록 시도
            try:
                import shutil as _sh
                tmp = Path(tempfile.gettempdir()) / f"answer_scan_dashboard_{int(time.time())}.py"
                _sh.copy2(str(db_path_obj), str(tmp))
                db_path_obj = tmp
                dashboard_py = str(db_path_obj)
                print(f"ℹ️ 권한 문제로 원본을 복사하여 임시 파일 사용: {dashboard_py}")
            except Exception as e:
                print(f"❌ 대시보드 파일 접근 및 임시 복사 실패: {e}")
                return
        except Exception as e:
            print(f"⚠️ 대시보드 파일 접근 중 예외: {e}")
            return

        # 실행 시 PATH 의존성을 피하기 위해 현 프로세스의 파이썬 실행기를 사용합니다.
        # frozen 상태라면 sys.executable은 exe 경로를 가리키지만
        # '-m streamlit' 방식으로 호출하면 배포 환경의 python을 사용하도록 보장할 수 있습니다.
        python_exec = sys.executable

        cmd = [
            python_exec,
            "-m",
            "streamlit",
            "run",
            dashboard_py,
            "--",
            f"--output_dir={resolved_output}",
            f"--base_dir={resolved_base}",
        ]
        if resolved_default:
            cmd.append(f"--default_result={resolved_default}")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["ANSWER_SCAN_OUTPUT_DIR"] = resolved_output
        env["ANSWER_SCAN_BASE_DIR"] = resolved_base
        env["ANSWER_SCAN_SELECTION_ROOT"] = str(sel_path)
        if resolved_default:
            env["ANSWER_SCAN_DEFAULT_RESULT"] = resolved_default

    # Windows에서는 'start'로 실행하면 부모 프로세스가 종료되어도
    # 자식 프로세스가 계속 실행되는 분리(detach) 프로세스가 생성됩니다.
    # 분리는 사용자가 명시적으로 --detach를 지정한 경우에만 수행합니다.
    # frozen(exe) 상태에서 자동으로 분리하면 부모 exe가 바로 종료되므로
    # 기본 동작은 분리하지 않고 부모 프로세스를 유지하여 exe가 사용자가
    # 닫을 때까지 계속 실행되도록 합니다.
        if os.name == 'nt' and args.detach:
            with dashboard_lock:
                if dashboard_state["started"]:
                    return
                cmdline = [
                    "cmd",
                    "/k",
                    "start",
                    "",
                    python_exec,
                    "-m",
                    "streamlit",
                    "run",
                    dashboard_py,
                    "--",
                    f"--output_dir={resolved_output}",
                    f"--base_dir={resolved_base}",
                ]
                if resolved_default:
                    cmdline.append(f"--default_result={resolved_default}")
                try:
                    subprocess.Popen(
                        cmdline,
                        cwd=os.path.dirname(os.path.dirname(__file__)),
                        env=env,
                    )
                    dashboard_state["started"] = True
                except Exception as e:
                    print(f"❌ 대시보드 분리 실행 실패: {e}")
            return

        with dashboard_lock:
            if dashboard_state["started"]:
                return
            cwd = os.path.dirname(os.path.dirname(__file__))
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
            except Exception as launch_exc:
                print(f"❌ 대시보드 실행 오류: {launch_exc}")
                return
            dashboard_state["started"] = True
            dashboard_state["proc"] = proc

        proc = dashboard_state.get("proc")
        if proc is not None:
            _wait_streamlit_ready_and_report(proc, timeout=90)

    def _scope_complete(scope, index, total, destination):
        if first_scope_event.is_set():
            return
        try:
            resolved_destination = str(Path(destination).resolve())
        except Exception:
            resolved_destination = str(destination)
        base_dir_resolved = _resolve_top_level_base(str(scope.source_dir.parent))
        first_scope_info["output_dir"] = resolved_destination
        first_scope_info["default_result"] = resolved_destination
        first_scope_info["base_dir"] = base_dir_resolved
        first_scope_event.set()

    def _worker():
        try:
            summary = run_scoped_pipeline(
                sel,
                effective_output_dir,
                config=cfg,
                recursive=args.recursive,
                progress_callback=progress_printer_silent,
                scope_complete_callback=_scope_complete,
            )
            _run_summary["summary"] = summary
        except Exception as worker_exc:
            _run_summary["error"] = worker_exc
            raise

    def _dashboard_waiter():
        first_scope_event.wait()
        info = first_scope_info.copy()
        output_dir = info.get("output_dir")
        if not output_dir:
            return
        base_dir = info.get("base_dir")
        default_result = info.get("default_result") or output_dir
        _maybe_launch_dashboard("first-scope", output_dir, base_dir, default_result)

    threading.Thread(target=_dashboard_waiter, daemon=True).start()

    analysis_start = time.time()
    analysis_duration: Optional[float] = None

    try:
        import tkinter as tk
        from tkinter import ttk

        class ToastToast:
            def __init__(self, shared_state, worker_thread, width=480, height=140):
                self.shared = shared_state
                self.worker = worker_thread
                self.width = width
                self.height = height
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

        # 파이프라인 스레드 시작 (토스트 창용 - 조용한 모드)
        pipeline_thread = threading.Thread(target=_worker, daemon=True)
        pipeline_thread.start()

        # 즉시 대시보드를 띄워 진행상황을 실시간으로 확인할 수 있게 합니다.
        try:
            # 가능한 한 안전한 기본값을 넘깁니다. 실제 결과는 나중에 갱신될 수 있습니다.
            _maybe_launch_dashboard("start", effective_output_dir, str(sel_path), effective_output_dir)
        except Exception:
            # 실패해도 진행은 계속됩니다.
            pass

    # 토스트 창을 메인 스레드에서 실행 (창이 닫힐 때까지 블로킹)
        try:
            toast = ToastToast(_progress_state, pipeline_thread)
            toast.run()
            print("\n✅ 분석 완료!")
        except Exception as e:
            print(f"토스트 창 실행 실패: {e}")
            # 실패하면 블록킹 방식으로 대체 실행
            pipeline_thread.join()
            print("\n✅ 분석 완료!")

        pipeline_thread.join()
        analysis_duration = time.time() - analysis_start
        if "error" in _run_summary:
            raise _run_summary["error"]
        # 파이프라인이 끝난 직후, 확인용으로 결과 ZIP(개별/aggregate)을 생성 시도합니다.
        try:
            summary_after = _run_summary.get("summary")
            if summary_after:
                rp = summary_after.get("result_paths") or []
                try:
                    from detector_pipeline import create_aggregate_result_zip, create_result_zip_for_dir
                except Exception:
                    try:
                        from .detector_pipeline import create_aggregate_result_zip, create_result_zip_for_dir
                    except Exception:
                        create_aggregate_result_zip = None
                        create_result_zip_for_dir = None
                # per-result zip
                if create_result_zip_for_dir and rp:
                    for p in rp:
                        try:
                            create_result_zip_for_dir(str(p))
                        except Exception:
                            pass
                # aggregate ZIP 생성 처리
                # 동작 요약(한국어):
                # - 목적: 사용자의 요구대로 "분석한 폴더의 상위 폴더"에 단 하나의 '총_결과_*.zip'을 생성합니다.
                #   예: 결과 폴더들이 'C:\\answer - 복사본\\인문계\\1교시\\11001_결과' 등이라면,
                #   공통 부모(base) = 'C:\\answer - 복사본\\인문계\\1교시'이며,
                #   이 블록은 그 상위 폴더인 'C:\\answer - 복사본\\인문계'에 ZIP을 생성합니다.
                # - 설계 이유: 대시보드 업로드나 사용자가 폴더 구조를 유지한 채 총 결과를 상위 레벨에서 일괄 확인할 수 있도록.
                # - 구현 세부: `rp` 목록에서 공통 부모(base)를 계산하고, 그 부모의 상위 폴더를 target으로 지정하여
                #   create_aggregate_result_zip(base, target_dir=target_parent) 한 번만 호출합니다.
                if create_aggregate_result_zip and rp:
                    try:
                        # *_결과 폴더들이 모여 있는 분석 폴더(예: ...\\1교시)
                        base_dir = os.path.commonpath(rp)
                    except Exception:
                        # 공통 경로 계산 실패 시 폴백: 현재의 출력 디렉터리 사용
                        base_dir = effective_output_dir

                    try:
                        base_path = Path(base_dir).resolve()
                    except Exception:
                        base_path = Path(base_dir)

                    # 목표 생성 위치: 분석 폴더(base_path)의 상위 폴더 (예: ...\\인문계)
                    try:
                        target_parent = str(base_path.parent)
                    except Exception:
                        target_parent = os.path.dirname(str(base_path))

                    try:
                        os.makedirs(target_parent, exist_ok=True)
                    except Exception:
                        # 디렉터리 생성 실패 시에도 계속 진행 (create_aggregate_result_zip 내부에서 다시 시도함)
                        pass

                    try:
                        # 한 번만 aggregate 생성 요청을 보냅니다. 내부에서 동일 이름이 이미 있으면 재사용합니다.
                        create_aggregate_result_zip(str(base_path), target_dir=target_parent)
                    except Exception:
                        # 실패 시 무시하고 넘어갑니다.
                        pass
        except Exception:
            pass
    except Exception:
        # tkinter가 없거나 실패 시 기존 동기 호출로 폴백 (콘솔 출력 모드)
        summary = run_scoped_pipeline(
            sel,
            effective_output_dir,
            config=cfg,
            recursive=args.recursive,
            progress_callback=progress_printer_console,
            scope_complete_callback=_scope_complete,
        )
        _run_summary["summary"] = summary
        analysis_duration = time.time() - analysis_start
        print("\n✅ 분석 완료!")

    summary = _run_summary.get("summary")
    summary_zip_paths: List[str] = []
    if summary:
        _print_run_summary(summary, analysis_duration)
        # scope_runner에서 이미 ZIP 경로를 넘겨주었다면 이를 수집합니다.
        try:
            summary_zip_paths = [str(Path(p).resolve()) for p in summary.get("result_zips", []) if p]
        except Exception:
            summary_zip_paths = [p for p in summary.get("result_zips", []) if p]
    elif analysis_duration is not None:
        _print_run_summary(None, analysis_duration)

    dashboard_output_dir = effective_output_dir
    base_candidate = sel_path if multi_scope_mode else (Path(effective_output_dir).parent if Path(effective_output_dir).parent != Path(effective_output_dir) else Path(effective_output_dir))
    dashboard_base_dir = _resolve_top_level_base(str(base_candidate))
    if summary:
        raw_paths = summary.get("result_paths") or []
        paths = [Path(p).resolve() for p in raw_paths]
        usable_paths = [p for p in paths if _result_dir_has_payload(p)]
        if usable_paths:
            dashboard_output_dir = str(usable_paths[0])
            default_result_path = str(usable_paths[0])
        elif paths:
            dashboard_output_dir = str(paths[0])
            default_result_path = str(paths[0])
            print("ℹ️ 결과 폴더가 생성되었지만 주요 리포트 파일이 보이지 않습니다. 대시보드에서 확인 후 적절한 폴더를 선택하세요.")
        if paths:
            try:
                common = Path(os.path.commonpath([str(p) for p in paths])).resolve()
                dashboard_base_dir = _resolve_top_level_base(str(common))
            except Exception:
                dashboard_base_dir = _resolve_top_level_base(str(sel_path))
        if not raw_paths and summary.get("mode") != "scoped" and default_result_path:
            dashboard_output_dir = default_result_path
        # 요약에 포함된 결과 폴더에 대해 ZIP 생성이 누락되었다면 보강합니다.
        ensured = _ensure_result_zips(raw_paths)
        if ensured:
            summary_zip_paths.extend([p for p in ensured if p])
    if not default_result_path and os.path.isdir(effective_output_dir):
        default_result_path = effective_output_dir
        dashboard_output_dir = effective_output_dir
    if not dashboard_base_dir:
        candidate = dashboard_output_dir if os.path.isdir(dashboard_output_dir) else effective_output_dir
        dashboard_base_dir = _resolve_top_level_base(candidate)
    else:
        dashboard_base_dir = _resolve_top_level_base(dashboard_base_dir)

    if not first_scope_info.get("output_dir") and default_result_path:
        first_scope_info["output_dir"] = dashboard_output_dir
        first_scope_info["default_result"] = default_result_path
        first_scope_info["base_dir"] = dashboard_base_dir

    # 대시보드/외부 도구가 가장 최근 ZIP 경로를 참조할 수 있도록 환경 변수에 기록합니다.
    if summary_zip_paths:
        # 최신 생성 ZIP을 우선으로 정렬 (수정 시간 기준)
        try:
            summary_zip_paths = sorted(set(summary_zip_paths), key=lambda p: Path(p).stat().st_mtime, reverse=True)
        except Exception:
            summary_zip_paths = list(dict.fromkeys(summary_zip_paths))
        os.environ["ANSWER_SCAN_RESULT_ZIPS"] = os.pathsep.join(summary_zip_paths)

    _maybe_launch_dashboard("summary", dashboard_output_dir, dashboard_base_dir, default_result_path)
    if not first_scope_event.is_set():
        first_scope_event.set()

    # 추가 진단: 실행 환경 및 대시보드 프로세스 상태를 출력하여
    # 즉시 종료되는 원인을 확인할 수 있게 합니다.
    try:
        is_frozen = getattr(sys, "frozen", False)
    except Exception:
        is_frozen = False
    try:
        proc = dashboard_state.get("proc") if isinstance(dashboard_state, dict) else None
    except Exception:
        proc = None
    try:
        # 기록은 로거에 위임: 기본 핸들러(콘솔 출력)는 로깅 레벨에 따라 출력 여부가 결정됩니다.
        logging.debug("sys.frozen=%s", is_frozen)
        logging.debug("dashboard_state keys=%s", (list(dashboard_state.keys()) if isinstance(dashboard_state, dict) else type(dashboard_state)))
    except Exception:
        pass
    try:
        # 상태 정보도 로거에 기록합니다(기본 동작은 출력하지 않음).
        if proc is None:
            logging.debug("dashboard proc: None (대시보드가 분리되었거나 실행 실패)")
        else:
            try:
                pid = getattr(proc, 'pid', 'unknown')
            except Exception:
                pid = 'unknown'
            try:
                poll = proc.poll()
            except Exception:
                poll = 'err'
            logging.debug("dashboard proc PID=%s poll=%s", pid, poll)
    except Exception:
        pass

    # 인터랙티브 콘솔이면 엔터를 눌러 결과를 확인하도록 대기
    # 디버그 출력을 위한 대기 동작은 제거했습니다. 필요한 경우 로깅 레벨을 조정하여
    # debug 정보를 확인하세요.

    # frozen 상태(exe)로 실행 중인 경우:
    # 대시보드 서브프로세스가 실행되어 있으면 해당 프로세스가 종료될 때까지
    # 부모 exe를 유지합니다. 분리(detach) 방식으로 실행되어 subprocess 핸들이
    # 없을 경우에는 사용자가 직접 프로세스를 종료할 때까지 계속 실행합니다.
    try:
        if getattr(sys, "frozen", False):
            proc = dashboard_state.get("proc")
            # subprocess 핸들이 있으면 종료될 때까지 대기
            if proc is not None:
                try:
                    # PID와 상태를 출력하여 빠르게 종료되는 원인을 파악할 수 있게 함
                    try:
                        pid_info = f"(PID={getattr(proc, 'pid', 'unknown')})"
                    except Exception:
                        pid_info = "(PID=unknown)"
                    print(f"대시보드가 실행 중입니다. {pid_info} 대시보드를 닫을 때까지 프로그램을 종료하지 않습니다.")

                    # 대시보드 프로세스가 종료될 때까지 블록 대기
                    while proc.poll() is None:
                        time.sleep(1)

                    # 프로세스가 끝났다면 종료 코드와 남은 로그를 출력
                    exit_code = proc.poll()
                    print(f"대시보드 프로세스가 종료되었습니다. 종료 코드: {exit_code}")
                    try:
                        # 남은 stdout을 읽어 가능한 로그를 출력
                        if getattr(proc, 'stdout', None):
                            remaining = proc.stdout.read()
                            if remaining:
                                print("--- 대시보드 로그(종료 시점) ---")
                                print(remaining)
                                print("--- 로그 끝 ---")
                    except Exception:
                        pass

                    # 콘솔 환경이면 사용자의 확인을 기다려 즉시 종료되는 현상을 방지
                    try:
                        if sys.stdin and sys.stdin.isatty():
                            print("엔터를 눌러 프로그램을 종료하세요...")
                            try:
                                input()
                            except Exception:
                                time.sleep(1)
                    except Exception:
                        # 비인터랙티브 환경인 경우 잠깐 대기 후 종료
                        time.sleep(2)
                except KeyboardInterrupt:
                    # 콘솔에서 테스트할 때 Ctrl+C로 종료 허용
                    pass
            else:
                # subprocess 핸들이 없는 경우(분리 실행),
                # 사용자가 직접 프로세스를 종료할 때까지 프로그램을 유지
                try:
                    print("프로그램을 계속 실행합니다. 종료하려면 프로세스를 직접 종료하세요.")
                    # 인터랙티브 셸이라면 엔터로 종료할 수 있도록 안내
                    if sys.stdin and sys.stdin.isatty():
                        print("대기 중입니다. 엔터를 누르면 종료합니다.")
                        try:
                            input()
                        except Exception:
                            # 입력이 불가능하면 무한 대기
                            while True:
                                time.sleep(10)
                    else:
                        # 비인터랙티브(예: 더블클릭 실행)인 경우 안전하게 장시간 대기
                        while True:
                            time.sleep(10)
                except KeyboardInterrupt:
                    pass
    except Exception:
        # 모니터링 로직 실패 시에도 프로그램이 예기치 않게 종료되지 않도록 처리
        pass

if __name__ == "__main__":
    # Windows(PyInstaller) 멀티프로세싱 호환: 내부 포크 인자 처리
    try:
        from multiprocessing import freeze_support, set_start_method
        freeze_support()
        # Windows 기본은 'spawn'이지만, 다른 환경에서 실행될 가능성 대비
        try:
            set_start_method('spawn')
        except Exception:
            pass
    except Exception:
        pass
    main()