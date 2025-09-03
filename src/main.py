import os
import csv
from detector_pipeline import detect_pipeline

def save_report(results, output_csv):
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["파일1", "파일2", "유사도", "상태", "그룹ID"])
        writer.writerows(results)

def main():
    input_dir = "input_images"
    output_dir = "output"

    for sub in ["grouped", "ok"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    print("🔍 2단계 파이프라인 기반 중복/유사 그룹 탐지 시작...")

    results, groups = detect_pipeline(input_dir, output_dir)

    report_path = os.path.join(output_dir, "report.csv")
    save_report(results, report_path)

    print("✅ 완료! 결과는 report.csv 확인")

if __name__ == "__main__":
    main()
