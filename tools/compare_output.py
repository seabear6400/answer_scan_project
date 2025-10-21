import argparse
import csv
import glob
import os
from pathlib import Path
from typing import List


def _load_expected(csv_path: Path) -> List[str]:
    expect: List[str] = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                expect.append(row[0])
    return expect


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare report.csv entries with actual image files")
    parser.add_argument("result_dir", nargs="?", help="결과 폴더 경로 (images_summary.csv가 있는 폴더)")
    parser.add_argument("--log", dest="log_path", help="로그 파일 경로", default=None)
    args = parser.parse_args()

    if args.result_dir:
        result_dir = Path(args.result_dir).expanduser()
    else:
        env_dir = os.environ.get("ANSWER_SCAN_COMPARE_DIR")
        if not env_dir:
            print("❌ 결과 폴더를 지정하세요 (--result_dir 또는 ANSWER_SCAN_COMPARE_DIR)")
            return 1
        result_dir = Path(env_dir).expanduser()

    if not result_dir.exists():
        print(f"❌ 경로를 찾을 수 없습니다: {result_dir}")
        return 1

    csv_path = result_dir / 'images_summary.csv'
    if not csv_path.exists():
        print(f"❌ images_summary.csv가 없습니다: {csv_path}")
        return 1

    expect = _load_expected(csv_path)

    log_path = Path(args.log_path).expanduser() if args.log_path else Path(__file__).resolve().with_name('compare_output_result.txt')

    with log_path.open('w', encoding='utf-8') as log:
        log.write(f'expected count from csv: {len(expect)}\n')
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        found: List[str] = []
        for pattern in exts:
            found.extend(glob.glob(str(result_dir / '**' / pattern), recursive=True))
        log.write(f'found count by glob: {len(found)}\n')
        found_bns = [os.path.basename(p) for p in found]
        missing = [fn for fn in expect if fn not in found_bns]
        log.write(f'missing count: {len(missing)}\n')
        for m in missing:
            log.write(f'MISSING: {m}\n')

        from collections import Counter

        cnt = Counter(found_bns)
        dups = [k for k, v in cnt.items() if v > 1]
        log.write(f'duplicates count: {len(dups)}\n')
        if dups:
            log.write(f'duplicates sample: {dups[:10]}\n')
        extra = [bn for bn in found_bns if bn not in expect]
        log.write(f'extra in filesystem but not in csv: {len(extra)}\n')
        if extra:
            log.write(str(extra[:20]) + '\n')

    print('Wrote results to', log_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
