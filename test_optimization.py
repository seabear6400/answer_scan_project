#!/usr/bin/env python3
"""
데이터 크기에 따른 적응적 최적화 테스트 스크립트
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detector_pipeline import DetectorConfig, optimize_config_for_data_size
import torch

def test_optimization():
    """다양한 데이터 크기에 대한 최적화 테스트"""
    
    # 기본 설정
    base_config = DetectorConfig()
    
    test_cases = [
        (10, "소규모"),      # 매우 적은 데이터
        (30, "소규모"),      # 소규모 데이터  
        (100, "중간규모"),   # 중간 규모
        (300, "중간규모"),   # 중간 규모 상한
        (800, "대규모"),     # 대규모
        (2000, "대규모"),    # 매우 큰 규모
    ]
    
    print("=== 데이터 크기별 적응적 최적화 테스트 ===\n")
    print("GPU 사용 가능:", torch.cuda.is_available())
    print("CPU 코어 수:", os.cpu_count())
    print()
    
    for n_images, category in test_cases:
        print(f"🔍 {category} 데이터 테스트: {n_images}개 이미지")
        print("-" * 50)
        
        # 최적화 전
        print("최적화 전:")
        print(f"  batch_size: {base_config.batch_size}")
        print(f"  num_workers: {base_config.num_workers}")
        print(f"  embed_backend: {base_config.embed_backend}")
        print(f"  ann_backend: {base_config.ann_backend}")
        print(f"  k: {base_config.k}")
        print(f"  prefilter: {base_config.prefilter}")
        
        # 최적화 적용
        optimized = optimize_config_for_data_size(base_config, n_images)
        
        print("\n최적화 후:")
        print(f"  batch_size: {optimized.batch_size}")
        print(f"  num_workers: {optimized.num_workers}")
        print(f"  embed_backend: {optimized.embed_backend}")
        print(f"  ann_backend: {optimized.ann_backend}")
        print(f"  k: {optimized.k}")
        print(f"  prefilter: {optimized.prefilter}")
        
        # 변화된 항목 표시
        changes = []
        if base_config.batch_size != optimized.batch_size:
            changes.append(f"batch_size: {base_config.batch_size} → {optimized.batch_size}")
        if base_config.num_workers != optimized.num_workers:
            changes.append(f"num_workers: {base_config.num_workers} → {optimized.num_workers}")
        if base_config.embed_backend != optimized.embed_backend:
            changes.append(f"embed_backend: {base_config.embed_backend} → {optimized.embed_backend}")
        if base_config.ann_backend != optimized.ann_backend:
            changes.append(f"ann_backend: {base_config.ann_backend} → {optimized.ann_backend}")
        if base_config.k != optimized.k:
            changes.append(f"k: {base_config.k} → {optimized.k}")
        if base_config.prefilter != optimized.prefilter:
            changes.append(f"prefilter: {base_config.prefilter} → {optimized.prefilter}")
            
        if changes:
            print("\n📊 주요 변화:")
            for change in changes:
                print(f"  • {change}")
        else:
            print("\n📊 변화 없음")
            
        print("\n" + "="*60 + "\n")
    
    # 자동 최적화 비활성화 테스트
    print("🚫 자동 최적화 비활성화 테스트")
    print("-" * 50)
    no_opt_config = DetectorConfig(auto_optimize=False)
    no_opt_result = optimize_config_for_data_size(no_opt_config, 1000)
    
    print("원본 설정과 동일한지 확인:")
    print(f"  batch_size: {no_opt_config.batch_size} == {no_opt_result.batch_size} ? {no_opt_config.batch_size == no_opt_result.batch_size}")
    print(f"  ann_backend: {no_opt_config.ann_backend} == {no_opt_result.ann_backend} ? {no_opt_config.ann_backend == no_opt_result.ann_backend}")
    
    print("\n✅ 테스트 완료!")

def test_specific_scenarios():
    """특정 시나리오별 최적화 테스트"""
    print("\n=== 특정 시나리오 테스트 ===\n")
    
    scenarios = [
        {
            "name": "시험지 채점 (소량)",
            "n_images": 20,
            "description": "소규모 시험지 채점 - 빠른 시작이 중요"
        },
        {
            "name": "대량 문서 스캔",
            "n_images": 1500,
            "description": "대량 문서 처리 - 처리량 최적화 필요"
        },
        {
            "name": "실시간 처리",
            "n_images": 5,
            "description": "실시간 또는 즉석 처리 - 지연시간 최소화"
        }
    ]
    
    base_config = DetectorConfig()
    
    for scenario in scenarios:
        print(f"📋 시나리오: {scenario['name']}")
        print(f"   설명: {scenario['description']}")
        print(f"   이미지 수: {scenario['n_images']}개")
        
        optimized = optimize_config_for_data_size(base_config, scenario['n_images'])
        
        print("   최적화 결과:")
        print(f"     • 배치 크기: {optimized.batch_size} (메모리 효율성과 처리량 균형)")
        print(f"     • 워커 수: {optimized.num_workers} (병렬 처리 수준)")
        print(f"     • 백엔드: {optimized.ann_backend} (검색 알고리즘)")
        print(f"     • k 값: {optimized.k} (후보 개수)")
        
        # 예상 성능 특성
        if scenario['n_images'] < 50:
            print("     🚀 예상 특성: 빠른 시작, 낮은 메모리 사용")
        elif scenario['n_images'] < 500:
            print("     ⚖️ 예상 특성: 균형잡힌 성능")
        else:
            print("     💪 예상 특성: 높은 처리량, 최대 병렬화")
            
        print()

if __name__ == "__main__":
    test_optimization()
    test_specific_scenarios()