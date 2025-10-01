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
    if os.path.exists(ok_dir):
        for f in os.listdir(ok_dir)[:3]:  # 처음 3개만
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_paths.append(os.path.join(ok_dir, f))
    
    if not test_paths:
        print("❌ 테스트할 이미지가 없습니다.")
        return
    
    print(f"📁 테스트 이미지: {len(test_paths)}개")
    
    # ROI 설정
    roi_ratio = (0.15, 0.15, 0.85, 0.85)
    
    # DINOv2 데이터셋 테스트
    print("\n🔬 DINOv2 데이터셋 테스트:")
    try:
        ds_dinov2 = ImgDataset(test_paths, roi_ratio, "dinov2")
        
        for i, (tensor, path) in enumerate(ds_dinov2):
            print(f"  이미지 {i+1}: {tensor.shape} - {os.path.basename(path)}")
            
            # 예상 크기 검증
            if tensor.shape != (3, 224, 224):
                print(f"    ⚠️ 잘못된 크기! 예상: (3, 224, 224), 실제: {tensor.shape}")
            else:
                print(f"    ✅ 올바른 크기")
                
    except Exception as e:
        print(f"    ❌ DINOv2 데이터셋 오류: {e}")
    
    # ResNet18 데이터셋 테스트
    print("\n🔬 ResNet18 데이터셋 테스트:")
    try:
        ds_resnet = ImgDataset(test_paths, roi_ratio, "resnet18")
        
        for i, (tensor, path) in enumerate(ds_resnet):
            print(f"  이미지 {i+1}: {tensor.shape} - {os.path.basename(path)}")
            
            # 예상 크기 검증
            if tensor.shape != (3, 224, 224):
                print(f"    ⚠️ 잘못된 크기! 예상: (3, 224, 224), 실제: {tensor.shape}")
            else:
                print(f"    ✅ 올바른 크기")
                
    except Exception as e:
        print(f"    ❌ ResNet18 데이터셋 오류: {e}")
    
    print("\n✅ 데이터셋 테스트 완료!")

if __name__ == "__main__":
    test_dataset()