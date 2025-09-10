from detector_pipeline import detect_pipeline, DetectorConfig
from pathlib import Path
import sys

# workspace paths
ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / 'output' / 'ok'
OUT_DIR = ROOT / 'output_test'

print('IN_DIR=', IN_DIR)
print('OUT_DIR=', OUT_DIR)

cfg = DetectorConfig()
try:
    detect_pipeline(str(IN_DIR), str(OUT_DIR), config=cfg)
    print('Pipeline finished')
except Exception as e:
    print('Pipeline failed:', e)
    sys.exit(1)
