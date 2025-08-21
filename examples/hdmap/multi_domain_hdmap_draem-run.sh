#!/bin/bash
# nohup ./examples/hdmap/multi_domain_hdmap_draem-run.sh > /dev/null 2>&1 &
# pkill -f "multi_domain_hdmap_draem-training.py"

# DRAEM 병렬 실험 실행 스크립트
# 멀티 GPU를 활용하여 실험 조건을 병렬로 실행

AVAILABLE_GPUS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
EXPERIMENT_CONDITIONS=(
    "DRAEM_quick_3epochs"
    # "DRAEM_baseline_50epochs"
    # "DRAEM_lower_lr"
    # "DRAEM_higher_lr"
    # "DRAEM_adaptive_lr"
    # "DRAEM_gradient_clip_01"
    # "DRAEM_gradient_clip_05"
    # "DRAEM_warmup_cosine"
    # "DRAEM_weight_decay_001"
    # "DRAEM_adam_vs_adamw"
    # "DRAEM_dropout_01"
    # "DRAEM_combo_regularized"
)
NUM_EXPERIMENTS=${#EXPERIMENT_CONDITIONS[@]}

# 로그 디렉토리 생성 
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/draem/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

SCRIPT_PATH="examples/hdmap/multi_domain_hdmap_draem-training.py"

echo "=================================="
echo "🚀 DRAEM 병렬 실험 시작"
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
    cd /home/taewan.hwang/study/anomalib
    CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run python "${SCRIPT_PATH}" \
        --experiment_name "${EXP_NAME}" \
        --results_dir "${LOG_DIR}" \
        --log_level "INFO" \
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
echo ""
echo "💡 실행 상태 확인:"
echo "   - 로그 확인: tail -f ${LOG_DIR}/output_exp_*.log"
echo "   - GPU 사용률: nvidia-smi"
echo "   - 프로세스 확인: ps aux | grep python"

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
echo "🎉 모든 DRAEM 실험 완료!"
echo "   성공: ${SUCCESS_COUNT}/${NUM_EXPERIMENTS}"
echo "   실패: ${FAILED_COUNT}/${NUM_EXPERIMENTS}"
echo "   로그 디렉토리: ${LOG_DIR}"
echo "=================================="

# 결과 요약 생성
SUMMARY_FILE="${LOG_DIR}/experiment_summary.txt"
echo "DRAEM 병렬 실험 요약" > "${SUMMARY_FILE}"
echo "실행 시간: $(date)" >> "${SUMMARY_FILE}"
echo "성공: ${SUCCESS_COUNT}/${NUM_EXPERIMENTS}" >> "${SUMMARY_FILE}"
echo "실패: ${FAILED_COUNT}/${NUM_EXPERIMENTS}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "실험 조건별 결과:" >> "${SUMMARY_FILE}"

for i in "${!EXPERIMENT_CONDITIONS[@]}"; do
    GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    EXP_NAME=${EXPERIMENT_CONDITIONS[$i]}
    LOG_FILE="${LOG_DIR}/output_exp_${i}_gpu${GPU_ID}.log"
    
    if [ -f "${LOG_FILE}" ]; then
        if grep -q "✅ 실험 완료" "${LOG_FILE}"; then
            echo "✅ ${EXP_NAME} (GPU ${GPU_ID})" >> "${SUMMARY_FILE}"
        else
            echo "❌ ${EXP_NAME} (GPU ${GPU_ID})" >> "${SUMMARY_FILE}"
        fi
    else
        echo "❓ ${EXP_NAME} (GPU ${GPU_ID}) - 로그 파일 없음" >> "${SUMMARY_FILE}"
    fi
done

# 실패한 실험이 있으면 경고
if [ $FAILED_COUNT -gt 0 ]; then
    echo "⚠️  ${FAILED_COUNT}개 실험이 실패했습니다."
    echo "   로그 파일을 확인하세요: ${LOG_DIR}/"
    echo ""
    echo "❌ 실패한 실험들:"
    for i in "${!EXPERIMENT_CONDITIONS[@]}"; do
        GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
        EXP_NAME=${EXPERIMENT_CONDITIONS[$i]}
        LOG_FILE="${LOG_DIR}/output_exp_${i}_gpu${GPU_ID}.log"
        
        if [ -f "${LOG_FILE}" ] && ! grep -q "✅ 실험 완료" "${LOG_FILE}"; then
            echo "   - ${EXP_NAME} (GPU ${GPU_ID})"
            echo "     로그: ${LOG_FILE}"
        fi
    done
fi

echo ""
echo "📁 생성된 파일들:"
echo "   로그 파일: ${LOG_DIR}/output_exp_*.log"  
echo "   결과 파일: ${LOG_DIR}/result_*.json"
echo "   실험 폴더: ${LOG_DIR}/MultiDomainHDMAP/draem/*/"
echo "   체크포인트: ${LOG_DIR}/MultiDomainHDMAP/draem/*/tensorboard_logs/checkpoints/*.ckpt"
echo "   시각화: ${LOG_DIR}/MultiDomainHDMAP/draem/*/tensorboard_logs/visualize/"
echo "   요약 파일: ${SUMMARY_FILE}"
echo ""
echo "📊 결과 확인 명령어:"
echo "   cat ${SUMMARY_FILE}"
echo "   ls -la ${LOG_DIR}/"
echo "   find ${LOG_DIR} -name '*.json' | head -5"
