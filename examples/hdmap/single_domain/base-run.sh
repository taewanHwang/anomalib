#!/bin/bash

# Single Domain Anomaly Detection 통합 실험 병렬 실행 스크립트
# 
# 🚀 기본 사용법:
#   ./examples/hdmap/single_domain/base-run.sh               # 사용법 안내
#   ./examples/hdmap/single_domain/base-run.sh 0             # 특정 실험 (ID 0)
#   ./examples/hdmap/single_domain/base-run.sh 0,1,2         # 여러 실험 (ID 0,1,2)
#   ./examples/hdmap/single_domain/base-run.sh all           # 전체 실험 (멀티 GPU 자동 할당)
#
# 🔥 백그라운드 실행 (추천):
#   nohup ./examples/hdmap/single_domain/base-run.sh all > single_domain_training.log 2>&1 &
#   nohup ./examples/hdmap/single_domain/base-run.sh 2 > patchcore_test.log 2>&1 &
#
# 📊 실행 상태 확인:
#   tail -f single_domain_training.log                      # 메인 스크립트 로그 확인  
#   tail -f results/*/single_domain_*.log                   # 개별 실험 상세 로그 확인
#   tail -f results/*/training_detail.log                   # 실험별 훈련 상세 로그 확인
#   ps aux | grep base-run.sh                               # 스크립트 실행 여부
#   ps aux | grep base-training                             # 개별 실험 진행 상황
#   nvidia-smi                                              # GPU 사용 현황
#
# 🛑 실행 중단:
#   pkill -f "single_domain.*base-run.sh"                   # 메인 스크립트 종료
#   pkill -f "single_domain.*base-training.py"              # 모든 single-domain 실험 종료
# 
# 🖥️ GPU 설정: AVAILABLE_GPUS 배열을 수정하세요
# 
# 📋 Single-Domain 특징:
#   - 단일 domain(A)에서 훈련 → 동일 domain(A)에서 평가
#   - 전통적인 anomaly detection 성능 측정
#   - 결과: domain A에서의 AUROC + F1Score

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
PYTHON_SCRIPT="$SCRIPT_DIR/base-training.py"
CONFIG_FILE="$SCRIPT_DIR/base-exp_condition3.json"

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
    
    # 세션 타임스탬프 생성
    SESSION_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    echo "🕐 세션 Timestamp: $SESSION_TIMESTAMP"
    echo ""
    
    # 각 실험을 GPU에 자동 분배하여 병렬 실행
    for ((i=0; i<TOTAL_EXPERIMENTS; i++)); do
        # 사용 가능한 GPU 중에서 순환 할당
        GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
        
        # 실험 조건에서 실험 이름 추출
        EXPERIMENT_NAME=$("$PYTHON_CMD" -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    data = json.load(f)
print(data['experiment_conditions'][$i]['name'])
")
        
        # 실험 디렉터리 생성
        RESULTS_DIR="results/$SESSION_TIMESTAMP"
        EXPERIMENT_DIR="$RESULTS_DIR/${EXPERIMENT_NAME}_${SESSION_TIMESTAMP}"
        mkdir -p "$EXPERIMENT_DIR"
        
        # 로그 파일 경로
        TRAINING_LOG="$EXPERIMENT_DIR/training_detail.log"
        
        echo "🔬 실험 $i 시작 (GPU $GPU_ID)"
        echo "   📁 실험 디렉터리: $EXPERIMENT_DIR"
        echo "   📄 로그 파일: $TRAINING_LOG"
        
        # 백그라운드에서 실험 실행 (로그를 실험 디렉터리에 직접 저장)
        (
            cd "$PROJECT_ROOT"
            echo "🔬 [Exp$i-GPU$GPU_ID] 실험 시작" >> "$TRAINING_LOG"
            
            # Python 스크립트를 실험 디렉터리의 로그 파일로 실행
            CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_CMD" "$PYTHON_SCRIPT" \
                --config "$CONFIG_FILE" \
                --experiment-id "$i" \
                --gpu-id "$GPU_ID" \
                --experiment-dir "$EXPERIMENT_DIR" \
                >> "$TRAINING_LOG" 2>&1
            
            echo "✅ [Exp$i-GPU$GPU_ID] 실험 완료" >> "$TRAINING_LOG"
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