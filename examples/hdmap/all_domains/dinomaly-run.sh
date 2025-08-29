#!/bin/bash
# nohup ./examples/hdmap/all_domains/dinomaly-run.sh > /dev/null 2>&1 &
# pkill -f "dinomaly-run.sh"
# pkill -f "examples/hdmap/all_domains/dinomaly-training.py"

# Dinomaly All Domains 병렬 실험 실행 스크립트
# 멀티 GPU를 활용하여 실험 조건을 병렬로 실행

AVAILABLE_GPUS=(6 7 8 9 10 11 12 13 14 15)

SCRIPT_PATH="examples/hdmap/all_domains/dinomaly-training.py"

# Python 스크립트에서 실험 조건 개수 가져오기 (JSON 파일명은 Python에서 관리)
NUM_EXPERIMENTS=$(python "${SCRIPT_PATH}" --get-experiment-count)

# 로그 디렉토리 생성 
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/dinomaly/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "=================================="
echo "🚀 Dinomaly All Domains 병렬 실험 시작"
echo "=================================="
echo "📁 로그 디렉토리: ${LOG_DIR}"
echo "🖥️  사용 GPU: ${AVAILABLE_GPUS[*]}"
echo "🧪 실험 조건: ${NUM_EXPERIMENTS}개"
echo ""

# 실험 할당 및 실행
echo "📋 실험 할당:"
for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    echo "   GPU ${GPU_ID}: 실험 ${i}"
done
echo ""

echo "🚀 병렬 실험 시작..."

# 백그라운드로 모든 실험 실행
PIDS=()
for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    
    echo "[$(date +%H:%M:%S)] 시작: GPU ${GPU_ID} - 실험 ${i}"
    
    # 백그라운드로 실험 실행 (안정적인 방법)
    python "${SCRIPT_PATH}" \
        --gpu-id "${GPU_ID}" \
        --experiment-id "${i}" \
        --log-dir "${LOG_DIR}" \
        > "${LOG_DIR}/output_exp_${i}_gpu${GPU_ID}.log" 2>&1 &
    
    # PID 저장
    PID=$!
    PIDS+=($PID)
    echo "   PID: ${PID}"

    # GPU 간격을 두어 초기화 충돌 방지
    sleep 2
done

echo ""
echo "⏳ 모든 실험이 백그라운드에서 실행 중..."
echo "   실행 중인 PID: ${PIDS[*]}"

# 모든 프로세스 완료 대기
SUCCESS_COUNT=0
FAILED_COUNT=0

for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    
    echo "⏳ 대기 중: GPU ${GPU_ID} - 실험 ${i} (PID: ${PID})"
    
    # 프로세스 완료 대기
    wait $PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✅ 완료: GPU ${GPU_ID} - 실험 ${i}"
        ((SUCCESS_COUNT++))
    else
        echo "[$(date +%H:%M:%S)] ❌ 실패: GPU ${GPU_ID} - 실험 ${i} (종료 코드: ${EXIT_CODE})"
        ((FAILED_COUNT++))
    fi
done

echo ""
echo "=================================="
echo "🎉 모든 Dinomaly All Domains 실험 완료!"
echo "   성공: ${SUCCESS_COUNT}/${NUM_EXPERIMENTS}"
echo "   실패: ${FAILED_COUNT}/${NUM_EXPERIMENTS}"
echo "   로그 디렉토리: ${LOG_DIR}"
echo "=================================="
echo ""

# 실패한 실험이 있으면 경고
if [ $FAILED_COUNT -gt 0 ]; then
    echo "⚠️  ${FAILED_COUNT}개 실험이 실패했습니다."
    echo "   로그 파일을 확인하세요: ${LOG_DIR}/"
fi

echo ""
echo "📁 생성된 파일들:"
echo "   출력 로그: ${LOG_DIR}/output_exp_*_gpu*.log"
echo "   실험별 폴더: ${LOG_DIR}/AllDomainsHDMAP/dinomaly/*/"
echo "   체크포인트: ${LOG_DIR}/AllDomainsHDMAP/dinomaly/*/tensorboard_logs/checkpoints/"
echo "   JSON 결과: ${LOG_DIR}/AllDomainsHDMAP/dinomaly/*/tensorboard_logs/result_*.json"