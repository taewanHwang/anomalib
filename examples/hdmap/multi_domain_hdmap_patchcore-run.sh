#!/bin/bash
# nohup ./examples/hdmap/multi_domain_hdmap_patchcore-run.sh > /dev/null 2>&1 &
# pkill -f "multi_domain_hdmap_patchcore-run.sh"
# pkill -f "examples/hdmap/multi_domain_hdmap_patchcore-training.py"

# PatchCore 병렬 실험 실행 스크립트
# 멀티 GPU를 활용하여 실험 조건을 병렬로 실행

AVAILABLE_GPUS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

SCRIPT_PATH="examples/hdmap/multi_domain_hdmap_patchcore-training.py"

# Python 스크립트에서 실험 조건 개수 가져오기 (JSON 파일명은 Python에서 관리)
NUM_EXPERIMENTS=$(python "${SCRIPT_PATH}" --get-experiment-count)

# 로그 디렉토리 생성 
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/patchcore/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "=================================="
echo "🚀 PatchCore 병렬 실험 시작"
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

# 백그라운드로 모든 실험 시작
pids=()
for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    GPU_ID=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    
    echo "🎯 GPU ${GPU_ID}에서 실험 ${i} 시작..."
    
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
    # PatchCore 결과 파일은 깊은 경로에 저장됨 - find 명령어 사용
    RESULT_COUNT=$(find "${LOG_DIR}" -name "result_exp_$(printf "%02d" $i)_*_gpu*.json" -type f 2>/dev/null | wc -l)
    if [ ${RESULT_COUNT} -gt 0 ]; then
        echo "   ✅ 실험 ${i}: 성공"
    else
        echo "   ❌ 실험 ${i}: 실패 또는 미완료"
    fi
done

echo ""
echo "🔍 상세 결과 분석:"
echo "   python examples/hdmap/analyze_experiment_results.py --results-dir ${LOG_DIR}"
echo ""
echo "📈 TensorBoard 시각화:"
echo "   tensorboard --logdir ${LOG_DIR} --port 6006"
echo ""

# 실험 성공률 계산
SUCCESS_COUNT=0
for i in $(seq 0 $((NUM_EXPERIMENTS-1))); do
    # PatchCore 결과 파일은 깊은 경로에 저장됨 - find 명령어 사용
    RESULT_COUNT=$(find "${LOG_DIR}" -name "result_exp_$(printf "%02d" $i)_*_gpu*.json" -type f 2>/dev/null | wc -l)
    if [ ${RESULT_COUNT} -gt 0 ]; then
        ((SUCCESS_COUNT++))
    fi
done

echo "📈 실험 완료 통계:"
echo "   전체 실험: ${NUM_EXPERIMENTS}개"
echo "   성공: ${SUCCESS_COUNT}개"
echo "   실패: $((NUM_EXPERIMENTS - SUCCESS_COUNT))개"
echo "   성공률: $(echo "scale=1; ${SUCCESS_COUNT} * 100 / ${NUM_EXPERIMENTS}" | bc -l)%"

if [ ${SUCCESS_COUNT} -eq ${NUM_EXPERIMENTS} ]; then
    echo ""
    echo "🎊 모든 실험이 성공적으로 완료되었습니다!"
elif [ ${SUCCESS_COUNT} -gt 0 ]; then
    echo ""
    echo "⚠️  일부 실험이 실패했습니다. 로그를 확인해주세요."
else
    echo ""
    echo "❌ 모든 실험이 실패했습니다. 설정을 확인해주세요."
fi

echo ""
echo "🏁 PatchCore 병렬 실험 완료!"
