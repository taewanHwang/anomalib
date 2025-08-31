#!/bin/bash

# Base Single Domain Anomaly Detection Training Script
# 
# 🚀 기본 사용법:
#   ./examples/hdmap/single_domain/base-run.sh          # 단일 실험 (ID 0)
#   ./examples/hdmap/single_domain/base-run.sh 5        # 특정 실험 (ID 5)
#   ./examples/hdmap/single_domain/base-run.sh all      # 전체 실험 (멀티 GPU 자동 할당)
#
# 🔥 백그라운드 실행 (추천):
#   nohup ./examples/hdmap/single_domain/base-run.sh all > training.log 2>&1 &
#   nohup ./examples/hdmap/single_domain/base-run.sh 4 > exp4.log 2>&1 &
#
# 📊 실행 상태 확인:
#   tail -f training.log                               # 로그 실시간 확인
#   ps aux | grep base-run.sh                         # 스크립트 실행 여부
#   ps aux | grep base-training                       # 개별 실험 진행 상황
#   nvidia-smi                                        # GPU 사용 현황
#
# 🛑 실행 중단:
#   pkill -f base-run.sh                              # 메인 스크립트 종료
#   pkill -f base-training.py                         # 모든 실험 프로세스 종료
# 
# 🖥️ GPU 설정: AVAILABLE_GPUS 배열을 수정하세요

set -e

# 스크립트 위치 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "🚀 Base Single Domain 실험 시작"
echo "📁 프로젝트: ${PROJECT_ROOT}"

# Python 환경 (.venv 사용)
PYTHON_CMD="$PROJECT_ROOT/.venv/bin/python"
if [ ! -f "$PYTHON_CMD" ]; then
    echo "❌ .venv가 없습니다: $PYTHON_CMD"
    exit 1
fi

# 사용 가능한 GPU 리스트
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

# 기본 설정
CONFIG_FILE="$SCRIPT_DIR/base-exp_condition1.json"
PYTHON_SCRIPT="$SCRIPT_DIR/base-training.py"

# 인자 처리
MODE=${1:-0}

if [ "$MODE" = "all" ]; then
    echo "🔥 전체 실험 멀티 GPU 실행"
    echo "🖥️ 사용 가능 GPU: ${AVAILABLE_GPUS[*]}"
    
    # 총 실험 수 확인
    TOTAL_EXPERIMENTS=$("$PYTHON_CMD" -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    data = json.load(f)
print(len(data['experiment_conditions']))
")
    
    echo "📋 총 실험 수: $TOTAL_EXPERIMENTS"
    echo "🖥️ 사용할 GPU 수: ${#AVAILABLE_GPUS[@]}개"
    
    # 각 실험을 GPU에 자동 분배하여 병렬 실행
    for ((i=0; i<TOTAL_EXPERIMENTS; i++)); do
        # 사용 가능한 GPU 중에서 순환 할당
        GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
        
        echo "🔬 실험 $i 시작 (GPU $GPU_ID)"
        
        # 백그라운드에서 실험 실행
        (
            cd "$PROJECT_ROOT"
            CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_CMD" "$PYTHON_SCRIPT" \
                --config "$CONFIG_FILE" \
                --experiment-id "$i" \
                --gpu-id "$GPU_ID"
        ) &
        
        # 모든 GPU가 사용 중이면 대기
        if (( (i + 1) % ${#AVAILABLE_GPUS[@]} == 0 )); then
            echo "⏳ GPU 세트 $((i / ${#AVAILABLE_GPUS[@]} + 1)) 완료 대기 중..."
            wait
        fi
    done
    
    # 남은 작업 대기
    echo "⏳ 모든 실험 완료 대기 중..."
    wait
    
    echo "🎉 전체 실험 완료!"
    
else
    # 단일 실험 실행
    EXPERIMENT_ID="$MODE"
    
    echo "🔧 설정:"
    echo "   실험 ID: $EXPERIMENT_ID"
    echo "   Python: $PYTHON_CMD"
    
    cd "$PROJECT_ROOT"
    "$PYTHON_CMD" "$PYTHON_SCRIPT" \
        --config "$CONFIG_FILE" \
        --experiment-id "$EXPERIMENT_ID"
    
    echo "✅ 실험 완료!"
fi