#!/bin/bash

# DRAEM-SevNet 병렬 실험 실행 스크립트
# 멀티 GPU를 활용하여 실험 조건을 병렬로 실행

# 설정
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7 8 9 10)
EXPERIMENT_CONDITIONS=(
    "single_scale_simple_avg"
    "multi_scale_simple_avg" 
    "single_scale_weighted_avg"
    "single_scale_maximum"
    "single_scale_smoothl1"
    "single_scale_weight_0p3"
    "single_scale_weight_0p7"
    "single_scale_large_patch"
    "single_scale_landscape_patch"
)
NUM_EXPERIMENTS=${#EXPERIMENT_CONDITIONS[@]}

# 로그 디렉토리 생성 (results 폴더로 통합)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/exp_logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

SCRIPT_PATH="examples/hdmap/multi_domain_hdmap_draem_sevnet_training.py"

echo "=================================="
echo "🚀 DRAEM-SevNet 병렬 실험 시작"
echo "=================================="
echo "📁 로그 디렉토리: ${LOG_DIR}"
echo "🖥️  사용 GPU: ${AVAILABLE_GPUS[*]}"
echo "🧪 실험 조건: ${NUM_EXPERIMENTS}개"
echo ""

# 실험 할당 및 실행
echo "📋 실험 할당:"
for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    EXP_NAME=${EXPERIMENT_CONDITIONS[$i]}
    echo "   GPU ${GPU_ID}: 실험 ${i} - ${EXP_NAME}"
done
echo ""

echo "🚀 병렬 실험 시작..."

# 백그라운드로 모든 실험 실행
PIDS=()
for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    EXP_NAME=${EXPERIMENT_CONDITIONS[$i]}
    
    echo "[$(date +%H:%M:%S)] 시작: GPU ${GPU_ID} - ${EXP_NAME}"
    
    # 백그라운드로 실험 실행
    cd /home/disk5/taewan/study/anomalib
    uv run "${SCRIPT_PATH}" \
        --gpu-id "${GPU_ID}" \
        --experiment-id "${i}" \
        --log-dir "${LOG_DIR}" \
        > "${LOG_DIR}/output_exp_${i}_gpu${GPU_ID}.log" 2>&1 &
    
    # PID 저장
    PID=$!
    PIDS+=($PID)
    echo "   PID: ${PID}"
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
    EXP_NAME=${EXPERIMENT_CONDITIONS[$i]}
    
    echo "⏳ 대기 중: GPU ${GPU_ID} - ${EXP_NAME} (PID: ${PID})"
    
    # 프로세스 완료 대기
    wait $PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✅ 완료: GPU ${GPU_ID} - ${EXP_NAME}"
        ((SUCCESS_COUNT++))
    else
        echo "[$(date +%H:%M:%S)] ❌ 실패: GPU ${GPU_ID} - ${EXP_NAME} (종료 코드: ${EXIT_CODE})"
        ((FAILED_COUNT++))
    fi
done

echo ""
echo "=================================="
echo "🎉 모든 실험 완료!"
echo "   성공: ${SUCCESS_COUNT}/${NUM_EXPERIMENTS}"
echo "   실패: ${FAILED_COUNT}/${NUM_EXPERIMENTS}"
echo "   로그 디렉토리: ${LOG_DIR}"
echo "=================================="

# 실패한 실험이 있으면 경고
if [ $FAILED_COUNT -gt 0 ]; then
    echo "⚠️  ${FAILED_COUNT}개 실험이 실패했습니다."
    echo "   로그 파일을 확인하세요: ${LOG_DIR}/"
fi

echo ""
echo "📁 생성된 파일들:"
echo "   로그 파일: ${LOG_DIR}/*.log"
echo "   결과 파일: ${LOG_DIR}/*.json"
