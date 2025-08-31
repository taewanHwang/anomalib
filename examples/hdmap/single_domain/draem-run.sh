#!/bin/bash
# nohup ./examples/hdmap/single_domain/draem-run.sh > /dev/null 2>&1 &
# pkill -f "single_domain/draem-run.sh"
# pkill -f "examples/hdmap/single_domain/draem-training.py"

# DRAEM Single Domain 병렬 실험 실행 스크립트
# 멀티 GPU를 활용하여 각 도메인별 실험 조건을 병렬로 실행

AVAILABLE_GPUS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
EXPERIMENT_CONDITIONS=(
    "domainA_baseline"
    "domainA_quick_test"
    "domainA_optimized"
    "domainB_baseline"
    "domainB_quick_test"
    "domainB_optimized"
    "domainC_baseline"
    "domainC_quick_test"
    "domainC_optimized"
    "domainD_baseline"
    "domainD_quick_test"
    "domainD_optimized"
    "domainA_higher_lr"
    "domainA_lower_lr"
    "domainA_large_batch"
    "domainA_weight_decay"
)
NUM_EXPERIMENTS=${#EXPERIMENT_CONDITIONS[@]}

# 로그 디렉토리 생성 
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/draem/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

SCRIPT_PATH="examples/hdmap/single_domain/draem-training.py"

echo "=================================="
echo "🚀 DRAEM Single Domain 병렬 실험 시작"
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

# 백그라운드로 모든 실험 시작
pids=()
for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    EXP_NAME=${EXPERIMENT_CONDITIONS[$i]}
    
    echo "🎯 GPU ${GPU_ID}에서 실험 ${i} (${EXP_NAME}) 시작..."
    
    # 각 실험을 백그라운드로 실행
    nohup python ${SCRIPT_PATH} \
        --gpu-id ${GPU_ID} \
        --experiment-id ${i} \
        --log-dir "${LOG_DIR}" \
        > "${LOG_DIR}/output_exp_${i}_gpu${GPU_ID}.log" 2>&1 &
    
    # PID 저장
    pids+=($!)
    
    # GPU간 시작 간격 (GPU 초기화 충돌 방지)
    sleep 5
done

echo ""
echo "✅ 모든 실험이 백그라운드에서 시작되었습니다!"
echo "📊 실시간 모니터링:"
echo "   watch -n 10 'nvidia-smi'"
echo ""
echo "📄 개별 로그 확인:"
for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    echo "   tail -f ${LOG_DIR}/output_exp_${i}_gpu${GPU_ID}.log"
done
echo ""

# 모든 백그라운드 작업 완료 대기
echo "⏳ 모든 실험 완료 대기 중..."
for pid in ${pids[*]}; do
    wait $pid
    echo "✅ 실험 완료: PID $pid"
done

echo ""
echo "🎉 모든 실험이 완료되었습니다!"
echo "📁 결과 위치: ${LOG_DIR}"
echo ""

# 최종 결과 요약
echo "📊 실험 결과 요약:"
for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    EXP_NAME=${EXPERIMENT_CONDITIONS[$i]}
    RESULT_FILE="${LOG_DIR}/result_exp_$(printf "%02d" $i)_${EXP_NAME}_gpu*.json"
    if ls ${RESULT_FILE} 1> /dev/null 2>&1; then
        echo "   ✅ ${EXP_NAME}: 성공"
    else
        echo "   ❌ ${EXP_NAME}: 실패 또는 미완료"
    fi
done

echo ""
echo "🔍 다음 단계:"
echo "   1. 개별 결과 확인: ls ${LOG_DIR}/*.json"
echo "   2. 로그 분석: grep 'Image AUROC' ${LOG_DIR}/*.log"
echo "   3. 결과 시각화 확인: ls ${LOG_DIR}/*_visualization.png"
echo "   4. TensorBoard 로그: tensorboard --logdir ${LOG_DIR}/*/tensorboard_logs"
echo ""