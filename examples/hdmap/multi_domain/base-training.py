#!/usr/bin/env python3
"""Multi-Domain Anomaly Detection 통합 훈련 스크립트

이 스크립트는 Multi-Domain Anomaly Detection 모델들의 통합 훈련을 수행합니다.
Single domain과 달리 source domain에서 훈련하고 multiple target domains에서 평가합니다.

주요 기능:
- JSON 기반 실험 조건 관리
- MultiDomainAnomalyTrainer를 통한 통합 모델 훈련
- Source domain 훈련 + Target domains 평가
- 결과 경로: results/timestamp/experiment_name/

사용법:
    python examples/hdmap/multi_domain/base-training.py \\
        --config examples/hdmap/multi_domain/base-exp_condition1.json \\
        --experiment-id 0 \\고 
        --gpu-id 0

    python examples/hdmap/multi_domain/base-training.py \\
        --config examples/hdmap/multi_domain/base-exp_condition1.json \\
        --experiment-id 5 \\
        --gpu-id 1 \\
        --session-timestamp 20250831_120000

병렬 실행:
    ./examples/hdmap/multi_domain/base-run.sh all
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# GPU 설정 (import 전에 설정)
def setup_gpu(gpu_id):
    """GPU 설정"""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"🔧 GPU {gpu_id} 설정 완료")

# Multi-domain trainer import
from anomaly_trainer import MultiDomainAnomalyTrainer

# Experiment utilities import - 상위 디렉토리에서 import  
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiment_utils import (
    setup_warnings_filter,
    setup_experiment_logging,
    load_experiment_conditions
)

# 경고 메시지 필터링
setup_warnings_filter()
logging.getLogger("anomalib.visualization.image.item_visualizer").setLevel(logging.ERROR)
logging.getLogger("anomalib.visualization").setLevel(logging.ERROR)
logging.getLogger("anomalib.callbacks").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")


def load_experiment_config(config_path: str, experiment_id: int):
    """실험 설정 로드"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        experiment_conditions = data.get("experiment_conditions", [])
        
        if experiment_id >= len(experiment_conditions):
            raise IndexError(f"실험 ID {experiment_id}가 범위를 벗어났습니다. 총 실험 수: {len(experiment_conditions)}")
        
        condition = experiment_conditions[experiment_id]
        
        print(f"📋 실험 조건 로드 완료:")
        print(f"   ID: {experiment_id}")
        print(f"   이름: {condition['name']}")
        print(f"   설명: {condition['description']}")
        print(f"   모델: {condition['config']['model_type']}")
        print(f"   Source Domain: {condition['config']['source_domain']}")
        print(f"   Target Domains: {condition['config']['target_domains']}")
        
        return condition
        
    except Exception as e:
        raise Exception(f"실험 설정 로드 실패: {e}")


def setup_experiment_directory(session_timestamp: str, experiment_name: str):
    """실험 디렉터리 설정 (single domain과 동일한 구조)"""
    # results/timestamp/experiment_name_timestamp/
    experiment_dir_name = f"{experiment_name}_{session_timestamp}"
    experiment_dir = Path("results") / session_timestamp / experiment_dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 실험 디렉터리 생성: {experiment_dir}")
    return experiment_dir


def setup_logging(experiment_dir: Path, experiment_name: str, session_timestamp: str):
    """로깅 설정"""
    # Multi-domain 로그 파일 생성
    log_file = experiment_dir / f"multi_domain_{experiment_name}.log"
    
    logger = setup_experiment_logging(
        log_file_path=str(log_file),
        experiment_name=f"multi_domain_{experiment_name}"
    )
    
    # 상세 로그 파일도 생성 (single domain과 유사)
    detail_log_file = experiment_dir / "training_detail.log"
    
    print(f"📝 로그 파일 설정:")
    print(f"   메인 로그: {log_file}")
    print(f"   상세 로그: {detail_log_file}")
    
    return logger, detail_log_file


def run_single_experiment(
    condition: dict,
    session_timestamp: str,
    gpu_id: int = None,
    log_level: str = "INFO"
):
    """단일 multi-domain 실험 실행"""
    experiment_name = condition["name"]
    
    print(f"\n{'='*100}")
    print(f"🧪 Multi-Domain 실험 시작")
    print(f"🆔 실험 이름: {experiment_name}")
    print(f"📅 세션 타임스탬프: {session_timestamp}")
    print(f"💻 GPU ID: {gpu_id}")
    print(f"{'='*100}")
    
    try:
        # 1. GPU 설정
        setup_gpu(gpu_id)
        
        # 2. 실험 디렉터리 설정
        experiment_dir = setup_experiment_directory(session_timestamp, experiment_name)
        
        # 3. 로깅 설정
        logger, detail_log_file = setup_logging(experiment_dir, experiment_name, session_timestamp)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # 4. Multi-Domain Trainer 초기화
        print(f"\n🏗️ Multi-Domain Trainer 초기화 중...")
        trainer = MultiDomainAnomalyTrainer(
            config=condition["config"],
            experiment_name=experiment_name,
            session_timestamp=session_timestamp,
            experiment_dir=str(experiment_dir)
        )
        
        logger.info(f"🧪 Multi-Domain 실험 시작: {experiment_name}")
        logger.info(f"실험 설정: {condition}")
        logger.info(f"GPU ID: {gpu_id}")
        
        # 5. 실험 실행
        print(f"🚀 Multi-Domain 실험 실행 중...")
        result = trainer.run_experiment()
        
        # 6. 결과 확인
        if result.get("status") == "success":
            print(f"\n✅ Multi-Domain 실험 성공!")
            print(f"📊 Source Domain AUROC: {result.get('source_results', {}).get('test_image_AUROC', 'N/A')}")
            
            # Target domains 결과 출력
            target_results = result.get('target_results', {})
            if target_results:
                print(f"🎯 Target Domains 결과:")
                for domain, domain_result in target_results.items():
                    auroc = domain_result.get('test_image_AUROC', 'N/A')
                    print(f"   {domain}: {auroc}")
            
            logger.info(f"✅ Multi-Domain 실험 성공: {experiment_name}")
        else:
            print(f"\n❌ Multi-Domain 실험 실패!")
            error = result.get("error", "Unknown error")
            print(f"오류: {error}")
            logger.error(f"❌ Multi-Domain 실험 실패: {experiment_name}, 오류: {error}")
        
        print(f"\n📁 실험 결과 저장 경로: {experiment_dir}")
        logger.info(f"📁 실험 결과 저장 경로: {experiment_dir}")
        
        return result
        
    except Exception as e:
        error_msg = f"Multi-Domain 실험 실행 중 오류 발생: {e}"
        print(f"\n❌ {error_msg}")
        
        if 'logger' in locals():
            logger.error(error_msg)
        
        return {
            "status": "failed",
            "error": str(e),
            "experiment_name": experiment_name
        }


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Multi-Domain Anomaly Detection 통합 훈련 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python examples/hdmap/multi_domain/base-training.py \\
      --config examples/hdmap/multi_domain/base-exp_condition1.json \\
      --experiment-id 0 --gpu-id 0

  python examples/hdmap/multi_domain/base-training.py \\
      --config examples/hdmap/multi_domain/base-exp_condition1.json \\
      --experiment-id 5 --gpu-id 1 \\
      --session-timestamp 20250831_120000
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="실험 조건 JSON 파일 경로"
    )
    parser.add_argument(
        "--experiment-id", 
        type=int, 
        required=True,
        help="실험 조건 ID (0부터 시작)"
    )
    parser.add_argument(
        "--gpu-id", 
        type=int, 
        default=None,
        help="사용할 GPU ID (기본: 자동 선택)"
    )
    parser.add_argument(
        "--session-timestamp", 
        type=str, 
        default=None,
        help="세션 타임스탬프 (기본: 현재 시간)"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="로그 레벨 (기본: INFO)"
    )
    
    args = parser.parse_args()
    
    # 세션 타임스탬프 설정
    session_timestamp = args.session_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"🚀 Multi-Domain Anomaly Detection 통합 훈련 시작")
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️ 명령행 인수:")
    for key, value in vars(args).items():
        print(f"   {key}: {value}")
    
    try:
        # 실험 설정 로드
        print(f"\n📋 실험 설정 로드 중...")
        condition = load_experiment_config(args.config, args.experiment_id)
        
        # 실험 실행
        result = run_single_experiment(
            condition=condition,
            session_timestamp=session_timestamp,
            gpu_id=args.gpu_id,
            log_level=args.log_level
        )
        
        # 최종 결과 출력
        print(f"\n{'='*100}")
        print(f"🏁 Multi-Domain 실험 완료")
        print(f"📊 상태: {result['status']}")
        
        if result["status"] == "success":
            print(f"✅ {condition['name']} 실험이 성공적으로 완료되었습니다!")
        else:
            print(f"❌ {condition['name']} 실험이 실패했습니다.")
            print(f"💥 오류: {result.get('error', 'Unknown error')}")
        
        print(f"📅 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}")
        
        # Exit code 설정
        sys.exit(0 if result["status"] == "success" else 1)
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 실험 실행 중 예상치 못한 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()