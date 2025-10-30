import os
import shutil
from pathlib import Path
import sys

# Add repo root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

try:
    from detector_pipeline import create_aggregate_result_zip
except Exception as e:
    print('Import failed:', e)
    raise

base = Path.cwd() / 'tmp_agg_test'
if base.exists():
    shutil.rmtree(base)
base.mkdir(parents=True)

# create two result folders
r1 = base / 'AAA_결과'
r2 = base / 'BBB_결과'
for d in (r1, r2):
    (d / 'artifacts').mkdir(parents=True)
    (d / 'ok').mkdir(parents=True)
    # create dummy report
    with open(d / 'report.csv', 'w', encoding='utf-8') as f:
        f.write('파일,밀도\nimg1.jpg,0.5\n')
    # create dummy file
    with open(d / 'ok' / 'img1.jpg', 'wb') as f:
        f.write(b'JPEG')

print('base dir:', base)
zip_path = create_aggregate_result_zip(str(base), target_dir=str(r1 / 'artifacts'))
print('zip_path:', zip_path)
if zip_path:
    print('zip exists:', Path(zip_path).exists())
    print('artifact files:', list((r1 / 'artifacts').iterdir()))
else:
    print('No zip created')

# show any status files
for p in base.rglob('총_결과_status.txt'):
    print('status file:', p)
    print(p.read_text(encoding='utf-8'))
