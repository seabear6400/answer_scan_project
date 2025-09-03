import os
from detector_deep import detect_duplicates_deep
import csv

def save_report(results, output_csv):
    """CSV 리포트 저장"""
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["파일1", "파일2", "유사도", "상태", "비고"])
        writer.writerows(results)

def main():
    input_dir = "input_images"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("🔍 딥러닝 기반 중복 탐지 시작...")

    results, dup_groups = detect_duplicates_deep(
        input_dir=input_dir,
        threshold=0.99,   # 확실한 중복 기준
        suspect_low=0.95  # 의심 후보 기준
    )

    report_path = os.path.join(output_dir, "report.csv")
    save_report(results, report_path)

    print("✅ 처리 완료!")
    print(f"📄 결과 리포트: {report_path}")
    if dup_groups:
        print(f"⚠️ 중복 그룹 {len(dup_groups)}개 발견됨 (재스캔 필요)")
    else:
        print("👍 중복 없음 (모두 정상)")

if __name__ == "__main__":
    main()
