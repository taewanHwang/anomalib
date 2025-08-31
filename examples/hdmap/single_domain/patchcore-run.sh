#!/bin/bash
# nohup ./examples/hdmap/single_domain/patchcore-run.sh > /dev/null 2>&1 &
# pkill -f "single_domain/patchcore-run.sh"
# pkill -f "examples/hdmap/single_domain/patchcore-training.py"

# PatchCore Single Domain 병렬 실험 실행 스크립트
# 멀티 GPU를 활용하여 각 도메인별 실험 조건을 병렬로 실행

AVAILABLE_GPUS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
EXPERIMENT_CONDITIONS=(
    "domainA_baseline"
    "domainA_lightweight"
    "domainA_high_memory"
    "domainA_memory_efficient"
    "domainB_baseline"
    "domainB_lightweight"
    "domainB_high_memory"
    "domainC_baseline"
    "domainC_lightweight"
    "domainC_high_memory"
    "domainD_baseline"
    "domainD_lightweight"
    "domainD_high_memory"
    "domainA_resnet101"
    "domainA_single_layer"
    "domainA_more_neighbors"
    "domainA_triple_layers"
    "domainA_ultra_efficient"
)
NUM_EXPERIMENTS=${#EXPERIMENT_CONDITIONS[@]}

# 로그 디렉토리 생성 
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/patchcore/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

SCRIPT_PATH="examples/hdmap/single_domain/patchcore-training.py"

echo "=================================="
echo "🚀 PatchCore Single Domain 병렬 실험 시작"
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
    sleep 3  # PatchCore는 빠르므로 간격을 줄임
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
echo "   3. 메모리 뱅크 크기 확인: grep 'Memory Bank Size' ${LOG_DIR}/*.log"
echo "   4. 결과 시각화 확인: ls ${LOG_DIR}/*_visualization.png"
echo "   5. TensorBoard 로그: tensorboard --logdir ${LOG_DIR}/*/tensorboard_logs"
echo ""

echo "💡 성능 분석 팁:"
echo "   - PatchCore는 피팅 속도가 빠르므로 대부분의 실험이 빠르게 완료됩니다"
echo "   - coreset_sampling_ratio가 클수록 메모리 사용량이 증가하지만 성능이 향상될 수 있습니다"
echo "   - backbone 모델이 클수록 더 나은 특징 추출이 가능하지만 메모리 사용량이 증가합니다"
echo "   - num_neighbors가 클수록 더 정확한 anomaly score 계산이 가능합니다"
echo ""