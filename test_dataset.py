#!/usr/bin/env python3
"""
DINOv2 입력 크기 문제 테스트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from PIL import Image
import numpy as np
from detector_pipeline import ImgDataset

def test_dataset():
    print("🧪 ImgDataset 테스트")
    
    # 테스트 이미지 경로 (실제 존재하는 파일)
    test_paths = []
    ok_dir = "output/ok"
    #!/usr/bin/env python3
    """
    DINOv2 입력 크기 문제 테스트
    """

    import sys
    import os
    from pathlib import Path

    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

    import torch
    import numpy as np
    from typing import List, Optional

    from detector_pipeline import ImgDataset


    def _resolve_ok_dir() -> Optional[Path]:
        env_dir = os.environ.get("ANSWER_SCAN_TEST_OK_DIR")
        if env_dir:
            return Path(env_dir).expanduser()

        workspace_root = Path(__file__).resolve().parent
        for candidate in workspace_root.rglob("*_결과"):
            ok_dir = candidate / "ok"
            if ok_dir.is_dir():
                return ok_dir
        return None


    def test_dataset() -> None:
        print("🧪 ImgDataset 테스트")

        test_paths: List[str] = []
        ok_dir_path = _resolve_ok_dir()
        if ok_dir_path and ok_dir_path.exists():
            for f in os.listdir(ok_dir_path)[:3]:  # 처음 3개만
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_paths.append(str(ok_dir_path / f))

        if not test_paths:
            print("❌ 테스트할 이미지가 없습니다. 환경변수 ANSWER_SCAN_TEST_OK_DIR 또는 *_결과/ok 폴더를 확인하세요.")
            return

        print(f"📁 테스트 이미지: {len(test_paths)}개")

        roi_ratio = (0.15, 0.15, 0.85, 0.85)

        print("\n🔬 DINOv2 데이터셋 테스트:")
        try:
            ds_dinov2 = ImgDataset(test_paths, roi_ratio, "dinov2")

            for i, (tensor, path) in enumerate(ds_dinov2):
                print(f"  이미지 {i+1}: {tensor.shape} - {os.path.basename(path)}")

                if tensor.shape != (3, 224, 224):
                    print(f"    ⚠️ 잘못된 크기! 예상: (3, 224, 224), 실제: {tensor.shape}")
                else:
                    print("    ✅ 올바른 크기")

        except Exception as e:
            print(f"    ❌ DINOv2 데이터셋 오류: {e}")

        print("\n🔬 ResNet18 데이터셋 테스트:")
        try:
            ds_resnet = ImgDataset(test_paths, roi_ratio, "resnet18")

            for i, (tensor, path) in enumerate(ds_resnet):
                print(f"  이미지 {i+1}: {tensor.shape} - {os.path.basename(path)}")

                if tensor.shape != (3, 224, 224):
                    print(f"    ⚠️ 잘못된 크기! 예상: (3, 224, 224), 실제: {tensor.shape}")
                else:
                    print("    ✅ 올바른 크기")

        except Exception as e:
            print(f"    ❌ ResNet18 데이터셋 오류: {e}")

        print("\n✅ 데이터셋 테스트 완료!")


    if __name__ == "__main__":
        test_dataset()
