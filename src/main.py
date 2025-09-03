import os
from detector_pipeline import detect_pipeline

def main():
    input_dir = "input_images"
    output_dir = "output"

    for sub in ["grouped", "ok", "blank_answers"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    print("🔍 탐지 실행 중... (pHash → CNN)")
    results, groups = detect_pipeline(input_dir, output_dir)
    print("✅ 탐지 완료 → report.csv, report.parquet 생성됨.")

    # 자동으로 대시보드 열기
    import subprocess
    print("🌐 대시보드를 실행합니다... 브라우저에서 자동으로 열립니다.")
    subprocess.run(["python", "-m", "streamlit", "run", "src/dashboard.py"])

if __name__ == "__main__":
    main()
