from detector_pipeline import detect_pipeline, DetectorConfig
from pathlib import Path
from typing import Optional
import argparse
import os
import sys


def _resolve_input_dir(root: Path) -> Optional[Path]:
    env_dir = os.environ.get("ANSWER_SCAN_PIPELINE_IN_DIR")
    if env_dir:
        return Path(env_dir).expanduser()

    for candidate in root.rglob("*_결과"):
        ok_dir = candidate / "ok"
        if ok_dir.is_dir():
            return ok_dir
    return None


def _resolve_output_dir(root: Path, fallback_input: Optional[Path]) -> Path:
    env_dir = os.environ.get("ANSWER_SCAN_PIPELINE_OUT_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    if fallback_input is not None:
        return fallback_input.parent / "pipeline_test_output"
    return root / "pipeline_test_output"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run detector pipeline for quick smoke testing")
    parser.add_argument("input_dir", nargs="?", help="폴더 경로 (예: *_결과/ok)")
    parser.add_argument("output_dir", nargs="?", help="출력 경로. 생략 시 pipeline_test_output")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    in_dir = Path(args.input_dir).expanduser() if args.input_dir else _resolve_input_dir(root)
    if not in_dir or not in_dir.exists():
        print("❌ 입력 폴더를 찾지 못했습니다. --input_dir 인자 또는 ANSWER_SCAN_PIPELINE_IN_DIR 환경 변수를 지정하세요.")
        return 1

    out_dir = Path(args.output_dir).expanduser() if args.output_dir else _resolve_output_dir(root, in_dir)

    print('IN_DIR=', in_dir)
    print('OUT_DIR=', out_dir)

    cfg = DetectorConfig()
    try:
        detect_pipeline(str(in_dir), str(out_dir), config=cfg)
        print('Pipeline finished')
        return 0
    except Exception as e:
        print('Pipeline failed:', e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
