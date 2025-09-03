import csv

def save_report(results, dup_groups, output_csv):
    """CSV 리포트 저장"""
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["파일명", "상태", "재스캔 필요"])
        writer.writerows(results)

    print("📄 report.csv 저장 완료!")
    if dup_groups:
        print("중복 그룹 발견:")
        for g in dup_groups:
            print(g)
