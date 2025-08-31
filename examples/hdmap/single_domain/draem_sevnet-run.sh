#!/bin/bash
# nohup ./examples/hdmap/single_domain/draem_sevnet-run.sh > /dev/null 2>&1 &
# pkill -f "single_domain/draem_sevnet-run.sh"
# pkill -f "examples/hdmap/single_domain/draem_sevnet-training.py"

# DRAEM-SevNet Single Domain 병렬 실험 실행 스크립트
# 멀티 GPU를 활용하여 Domain A에서 DRAEM-SevNet 실험 조건을 병렬로 실행

AVAILABLE_GPUS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
EXPERIMENT_CONDITIONS=(
    "domainA_sevnet_optimal_multi_scale_landscape_severity25_patch_ratio_035"
    "domainA_sevnet_optimal_multi_scale_landscape_severity22"
    "domainA_sevnet_optimal_multi_scale_landscape_severity28"
    "domainA_sevnet_optimal_multi_scale_landscape_severity25_patch_ratio_05"
    "domainA_sevnet_optimal_size16_landscape_severity20_patch_ratio_04"
    "domainA_sevnet_optimal_size16_landscape_severity25"
    "domainA_sevnet_optimal_size14_landscape_severity20"
    "domainA_sevnet_optimal_size18_landscape_severity20"
    "domainA_sevnet_optimal_size8_landscape_severity25_patch_ratio_04"
    "domainA_sevnet_optimal_size8_landscape_severity50_patch_ratio_04"
    "domainA_sevnet_optimal_size12_landscape_severity20_patch_ratio_04"
    "domainA_sevnet_optimal_size12_landscape_severity25"
    "domainA_sevnet_optimal_multi_scale_landscape_severity25_patch_count_2"
    "domainA_sevnet_optimal_size16_landscape_severity20_patch_count_2"
    "domainA_sevnet_optimal_multi_scale_landscape_severity25_adam"
    "domainA_sevnet_optimal_size16_landscape_severity20_adam"
)
NUM_EXPERIMENTS=${#EXPERIMENT_CONDITIONS[@]}

# 로그 디렉토리 생성 
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/draem_sevnet/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

SCRIPT_PATH="examples/hdmap/single_domain/draem_sevnet-training.py"

echo "=================================="
echo "🚀 DRAEM-SevNet Single Domain 병렬 실험 시작"
echo "=================================="
echo "📁 로그 디렉토리: ${LOG_DIR}"
echo "🖥️  사용 GPU: ${AVAILABLE_GPUS[*]}"
echo "🧪 실험 조건: ${NUM_EXPERIMENTS}개"
echo "🎯 대상 도메인: Domain A (Single Domain)"
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
    sleep 8  # DRAEM-SevNet는 메모리 사용량이 크므로 간격을 늘림
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
echo "   3. 최고 성능 실험 확인: grep -l 'optimal_multi_scale_landscape_severity25_patch_ratio_035' ${LOG_DIR}/*.log"
echo "   4. 결과 시각화 확인: ls ${LOG_DIR}/*_visualization.png"
echo "   5. TensorBoard 로그: tensorboard --logdir ${LOG_DIR}/*/tensorboard_logs"
echo ""

echo "💡 성능 분석 팁:"
echo "   - DRAEM-SevNet는 DRAEM + Severity Head 조합으로 더 정밀한 이상 탐지가 가능합니다"
echo "   - Multi-Scale 모드는 다양한 해상도의 특징을 결합하여 성능을 향상시킵니다"
echo "   - Spatial-Aware pooling은 공간 정보를 보존하여 localization 성능을 개선합니다"
echo "   - Severity 값이 클수록 더 강한 이상 패턴을 생성하여 robust한 학습이 가능합니다"
echo "   - Patch ratio 범위는 이상 영역의 크기를 결정하며 도메인 특성에 맞게 조정 필요합니다"
echo ""

echo "🏆 최적 조건 후보:"
echo "   1. domainA_sevnet_optimal_multi_scale_landscape_severity25_patch_ratio_035 (최고 성능 검증됨)"
echo "   2. domainA_sevnet_optimal_multi_scale_landscape_severity22 (안정성 우수)"
echo "   3. domainA_sevnet_optimal_size16_landscape_severity20_patch_ratio_04 (효율성 우수)"
echo ""