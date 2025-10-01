#!/usr/bin/env python3
"""
GPU/CPU 수정사항 간단 테스트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detector_pipeline import diagnose_gpu, get_device_info

def main():
    print("🔧 수정된 GPU 진단 테스트")
    print("=" * 50)
    
    # 1. 빠른 GPU 진단
    print("\n1️⃣ 빠른 GPU 진단:")
    diagnose_gpu()
    
    # 2. 디바이스 정보 조회
    print("\n2️⃣ 디바이스 정보 조회:")
    device_info = get_device_info()
    print(f"결과: {device_info}")
    
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    main()