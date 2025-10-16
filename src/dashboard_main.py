import argparse
import os
import sys
import subprocess
import time


def parse_args():
    p = argparse.ArgumentParser(description="Answer Sheet QA — dashboard only (Streamlit)")
    p.add_argument("--output_dir", default="output", help="결과 폴더(대시보드가 읽을 경로)")
    p.add_argument("--port", type=int, default=8501, help="Streamlit 포트")
    p.add_argument("--browser", action="store_true", help="기본 브라우저로 자동 열기")
    p.add_argument("--detach", action="store_true", help="윈도우에서 새 창으로 분리 실행")
    return p.parse_args()


def main():
    args = parse_args()

    cmd = [sys.executable, "-m", "streamlit", "run", "src/dashboard.py", "--", f"--output_dir={args.output_dir}", f"--server.port={args.port}"]

    try:
        if args.detach and os.name == 'nt':
            # Windows: start in new window
            subprocess.Popen(["cmd", "/c", "start"] + cmd)
            print(f"✅ 대시보드 시작 (분리 창). 포트: {args.port}")
            return

        # 실행 및 스트림릿이 준비되면 반환
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
        ready_patterns = ("Local URL:", "Network URL:", "You can now view your Streamlit app", "Running on")
        t0 = time.time()
        timeout = 90
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if any(pat in line for pat in ready_patterns):
                elapsed = time.time() - t0
                print(f"✅ 대시보드 준비 완료 ({elapsed:.1f}s). 포트: {args.port}")
                if args.browser:
                    try:
                        import webbrowser
                        webbrowser.open(f"http://localhost:{args.port}")
                    except Exception:
                        pass
                proc.wait()
                return
            # 간단한 출력
            # print(f"[streamlit] {line}")
            if time.time() - t0 > timeout:
                print(f"⚠️ 대시보드 시작 대기시간 초과 ({timeout}s). 로그를 확인하세요.")
                return
    except KeyboardInterrupt:
        print("중단: 사용자가 취소했습니다.")
    except Exception as e:
        print(f"대시보드 실행 오류: {e}")


if __name__ == '__main__':
    main()
