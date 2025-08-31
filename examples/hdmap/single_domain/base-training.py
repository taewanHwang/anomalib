#!/usr/bin/env python3
"""
Base Single Domain Training Script for HDMAP Dataset

이 스크립트는 BaseAnomalyTrainer를 사용하여 모든 anomaly detection 모델의 
단일 도메인 실험을 통합 관리합니다.

지원 모델:
- DRAEM: Reconstruction + Anomaly Detection
- Dinomaly: Vision Transformer 기반 anomaly detection with DINOv2
- PatchCore: Memory bank 기반 few-shot anomaly detection  
- DRAEM-SevNet: Selective feature reconstruction

사용법:
    python examples/hdmap/single_domain/base-training.py --config base-exp_condition1.json
"""

import os
import torch
import argparse
from datetime import datetime
import sys

# 공통 유틸리티 함수들 import
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiment_utils import (
    setup_warnings_filter, 
    load_experiment_conditions
)

# BaseAnomalyTrainer import
from anomaly_trainer import BaseAnomalyTrainer

# 경고 메시지 비활성화
setup_warnings_filter()


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Base Single Domain 실험")
    parser.add_argument("--config", type=str, required=True, help="실험 조건 JSON 파일")
    parser.add_argument("--gpu-id", type=int, default=0, help="사용할 GPU ID")
    parser.add_argument("--experiment-id", type=int, default=None, help="특정 실험 조건 인덱스 (없으면 모든 실험 실행)")
    parser.add_argument("--log-dir", type=str, default=None, help="로그 저장 디렉토리")
    parser.add_argument("--experiment-dir", type=str, default=None, help="실험 디렉터리 (bash 스크립트에서 전달)")
    
    args = parser.parse_args()
    
    # GPU 설정
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"🖥️ GPU {args.gpu_id} 사용")
    
    # 경고 필터 설정
    setup_warnings_filter()
    
    # 실험 조건 로드
    conditions = load_experiment_conditions(args.config)
    
    # 세션 timestamp 생성
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🕐 세션 Timestamp: {session_timestamp}")
    
    # 실험 실행
    if args.experiment_id is not None:
        # 특정 실험만 실행
        if args.experiment_id >= len(conditions):
            print(f"❌ 잘못된 실험 ID: {args.experiment_id} (최대: {len(conditions)-1})")
            return
        
        condition = conditions[args.experiment_id]
        
        # bash 스크립트에서 실험 디렉터리가 전달된 경우 사용
        if args.experiment_dir:
            trainer = BaseAnomalyTrainer(condition["config"], condition["name"], session_timestamp, experiment_dir=args.experiment_dir)
        else:
            trainer = BaseAnomalyTrainer(condition["config"], condition["name"], session_timestamp)
        
        result = trainer.run_experiment()
        
        if "error" not in result:
            print(f"\n🎉 실험 성공!")
            if "results" in result and isinstance(result["results"], dict):
                print(f"   📊 최종 성과:")
                print(f"      Image AUROC: {result['results'].get('image_AUROC', 0):.4f}")
        else:
            print(f"\n💥 실험 실패: {result['error']}")
    else:
        # 모든 실험 실행
        successful_experiments = 0
        total_experiments = len(conditions)
        
        for i, condition in enumerate(conditions):
            print(f"\n{'='*60}")
            print(f"실험 {i+1}/{total_experiments}: {condition['name']}")
            print(f"설명: {condition['description']}")
            print(f"{'='*60}")
            
            trainer = BaseAnomalyTrainer(condition["config"], condition["name"], session_timestamp)
            result = trainer.run_experiment()
            
            if "error" not in result:
                successful_experiments += 1
                print(f"✅ 실험 {condition['name']} 성공")
            else:
                print(f"❌ 실험 {condition['name']} 실패: {result['error']}")
        
        # 최종 결과 요약
        print(f"\n{'='*60}")
        print(f"모든 실험 완료 - 세션: {session_timestamp}")
        print(f"성공: {successful_experiments}/{total_experiments}")
        print(f"실패: {total_experiments - successful_experiments}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()