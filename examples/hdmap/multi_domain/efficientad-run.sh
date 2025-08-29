#!/bin/bash
# nohup ./examples/hdmap/multi_domain_hdmap_efficientad-run.sh > /dev/null 2>&1 &
# pkill -f "multi_domain_hdmap_efficientad-run.sh"
# pkill -f "examples/hdmap/multi_domain_hdmap_efficientad-training.py"

# EfficientAD 병렬 실험 실행 스크립트
# 멀티 GPU를 활용하여 실험 조건을 병렬로 실행

# AVAILABLE_GPUS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
AVAILABLE_GPUS=(15)

SCRIPT_PATH="examples/hdmap/multi_domain_hdmap_efficientad-training.py"

# Python 스크립트에서 실험 조건 개수 가져오기 (JSON 파일명은 Python에서 관리)
# Python 환경 활성화 후 실행
source .venv/bin/activate
NUM_EXPERIMENTS=$(python "${SCRIPT_PATH}" --get-experiment-count)

# 로그 디렉토리 생성 
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/efficientad/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "=================================="
echo "🚀 EfficientAD 병렬 실험 시작"
echo "=================================="
echo "📁 로그 디렉토리: ${LOG_DIR}"
echo "🖥️  사용 GPU: ${AVAILABLE_GPUS[*]}"
echo "🧪 실험 조건: ${NUM_EXPERIMENTS}개"
echo ""

# GPU별 실험 할당
echo "📋 실험 할당:"
if [ ${NUM_EXPERIMENTS} -gt 0 ]; then
    for ((i=0; i<${NUM_EXPERIMENTS}; i++)); do
        gpu_idx=$((i % ${#AVAILABLE_GPUS[@]}))
        gpu_id=${AVAILABLE_GPUS[$gpu_idx]}
        echo "   GPU ${gpu_id}: 실험 ${i}"
    done
    echo ""
    
    # 병렬 실험 시작
    echo "🚀 병렬 실험 시작..."
    pids=()
    
    for ((i=0; i<${NUM_EXPERIMENTS}; i++)); do
        gpu_idx=$((i % ${#AVAILABLE_GPUS[@]}))
        gpu_id=${AVAILABLE_GPUS[$gpu_idx]}
        
        echo "🎯 GPU ${gpu_id}에서 실험 ${i} 시작..."
        
        # 개별 로그 파일과 실험 로그 파일 설정
        output_log="${LOG_DIR}/output_exp_${i}_gpu${gpu_id}.log"
        experiment_log="${LOG_DIR}/efficientad_experiment_$(date +"%Y%m%d_%H%M%S").log"
        
        # 환경 변수 설정하여 Python 스크립트가 로그 디렉터리를 알 수 있도록 함
        nohup bash -c "
            source .venv/bin/activate
            export EXPERIMENT_LOG_DIR='${LOG_DIR}'
            {
                echo '================================================================================'
                echo '🚀 EfficientAD 실험 시작: \$(python '${SCRIPT_PATH}' --get-experiment-count > /dev/null 2>&1 && python -c \"import json; data=json.load(open('examples/hdmap/multi_domain_hdmap_efficientad-exp_condition-test.json')); print(data['experiment_conditions'][${i}]['name'] if ${i} < len(data['experiment_conditions']) else 'unknown')\")'
                echo 'GPU ID: ${gpu_id} | 실험 ID: ${i}'
                echo '설명: \$(python -c \"import json; data=json.load(open('examples/hdmap/multi_domain_hdmap_efficientad-exp_condition-test.json')); print(data['experiment_conditions'][${i}]['description'] if ${i} < len(data['experiment_conditions']) else '실험 조건 없음')\")'
                echo '================================================================================'
                
                if python '${SCRIPT_PATH}' --gpu-id ${gpu_id} --experiment-id ${i} --results-dir '${LOG_DIR}'; then
                    echo '✅ 실험 성공!'
                    # 성공 시 결과 요약
                    if [ -f '${LOG_DIR}/result_exp_$(printf \"%02d\" ${i})_*_gpu${gpu_id}_*.json' ]; then
                        source_auroc=\$(python -c \"
import json, glob
files = glob.glob('${LOG_DIR}/result_exp_$(printf \"%02d\" ${i})_*_gpu${gpu_id}_*.json')
if files:
    with open(files[0]) as f:
        data = json.load(f)
        print(data.get('source_results', {}).get('test_image_AUROC', 'N/A'))
else:
    print('N/A')
\")
                        avg_auroc=\$(python -c \"
import json, glob
files = glob.glob('${LOG_DIR}/result_exp_$(printf \"%02d\" ${i})_*_gpu${gpu_id}_*.json')
if files:
    with open(files[0]) as f:
        data = json.load(f)
        print(f\\\"{data.get('avg_target_auroc', 0.0):.4f}\\\")
else:
    print('N/A')
\")
                        checkpoint=\$(python -c \"
import json, glob
files = glob.glob('${LOG_DIR}/result_exp_$(printf \"%02d\" ${i})_*_gpu${gpu_id}_*.json')
if files:
    with open(files[0]) as f:
        data = json.load(f)
        print(data.get('best_checkpoint', 'N/A'))
else:
    print('N/A')
\")
                        echo \"   Source Domain AUROC: \${source_auroc}\"
                        echo \"   Target Domains Avg AUROC: \${avg_auroc}\"
                        echo \"   체크포인트: \${checkpoint}\"
                        
                        # 훈련 정보 출력
                        training_info=\$(python -c \"
import json, glob
files = glob.glob('${LOG_DIR}/result_exp_$(printf \"%02d\" ${i})_*_gpu${gpu_id}_*.json')
if files:
    with open(files[0]) as f:
        data = json.load(f)
        info = data.get('training_info', {})
        print(f\\\"📊 학습 과정 정보:\\\")
        print(f\\\"   설정된 최대 에포크: {info.get('max_epochs_configured', 'N/A')}\\\")
        print(f\\\"   실제 학습 에포크: {info.get('last_trained_epoch', 'N/A')}\\\")
        print(f\\\"   총 학습 스텝: {info.get('total_steps', 'N/A')}\\\")
        print(f\\\"   Early Stopping 적용: {info.get('early_stopped', 'N/A')}\\\")
        print(f\\\"   최고 Validation AUROC: {info.get('best_val_auroc', 'N/A')}\\\")
        print(f\\\"   학습 완료 방식: {info.get('completion_description', 'N/A')}\\\")
\")
                        echo \"\${training_info}\"
                        
                        # Target domain별 성능 출력
                        echo \"🎯 Target Domain별 성능:\"
                        python -c \"
import json, glob
files = glob.glob('${LOG_DIR}/result_exp_$(printf \"%02d\" ${i})_*_gpu${gpu_id}_*.json')
if files:
    with open(files[0]) as f:
        data = json.load(f)
        for domain, metrics in data.get('target_results', {}).items():
            auroc = metrics.get('test_image_AUROC', 'N/A')
            print(f'   {domain}: {auroc}')
\"
                    fi
                    
                    # 결과 파일 경로
                    result_file=\$(find '${LOG_DIR}' -name \"result_exp_$(printf \"%02d\" ${i})_*_gpu${gpu_id}_*.json\" -type f | head -1)
                    if [ -n \"\${result_file}\" ]; then
                        echo \"📁 결과 파일: \${result_file}\"
                    fi
                    
                    # 실험 폴더 경로
                    exp_folder=\$(python -c \"
import json, glob
files = glob.glob('${LOG_DIR}/result_exp_$(printf \"%02d\" ${i})_*_gpu${gpu_id}_*.json')
if files:
    with open(files[0]) as f:
        data = json.load(f)
        print(data.get('experiment_path', 'N/A'))
\" 2>/dev/null || echo 'N/A')
                    if [ \"\${exp_folder}\" != \"N/A\" ]; then
                        echo \"📂 실험 폴더: \${exp_folder}\"
                    fi
                else
                    echo '❌ 실험 실패!'
                    echo \"   오류: \$?\"
                fi
                echo '✅ 실험 완료!'
                
                # GPU 메모리 정리
                python -c \"
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('🧹 GPU 메모리 정리 완료')
\"
                
            } 2>&1 | tee '${experiment_log}'
        " > "${output_log}" 2>&1 &
        
        pid=$!
        pids+=($pid)
        
        # GPU 메모리 충돌 방지를 위한 약간의 지연
        sleep 2
    done
    
    echo "✅ 모든 실험이 백그라운드에서 시작되었습니다!"
    echo "📊 실시간 모니터링:"
    echo "   watch -n 10 'nvidia-smi'"
    echo ""
    echo "📄 개별 로그 확인:"
    for ((i=0; i<${NUM_EXPERIMENTS}; i++)); do
        gpu_idx=$((i % ${#AVAILABLE_GPUS[@]}))
        gpu_id=${AVAILABLE_GPUS[$gpu_idx]}
        echo "   tail -f ${LOG_DIR}/output_exp_${i}_gpu${gpu_id}.log"
    done
    echo ""
    
    # 모든 프로세스 완료 대기
    echo "⏳ 모든 실험 완료 대기 중..."
    for pid in "${pids[@]}"; do
        wait $pid
        echo "✅ 실험 완료: PID $pid"
    done
    
    echo ""
    echo "🎉 모든 실험이 완료되었습니다!"
    echo "📁 결과 위치: ${LOG_DIR}"
    echo ""
    
    # 결과 요약
    echo "📊 실험 결과 요약:"
    successful_experiments=0
    failed_experiments=0
    
    for ((i=0; i<${NUM_EXPERIMENTS}; i++)); do
        gpu_idx=$((i % ${#AVAILABLE_GPUS[@]}))
        gpu_id=${AVAILABLE_GPUS[$gpu_idx]}
        
        # 결과 파일 존재 여부로 성공/실패 판단
        result_count=$(find "${LOG_DIR}" -name "result_exp_$(printf "%02d" $i)_*_gpu${gpu_id}.json" -type f 2>/dev/null | wc -l)
        
        if [ ${result_count} -gt 0 ]; then
            echo "   ✅ 실험 ${i}: 성공"
            ((successful_experiments++))
        else
            echo "   ❌ 실험 ${i}: 실패 또는 미완료"
            ((failed_experiments++))
        fi
    done
    
    echo ""
    echo "🔍 상세 결과 분석:"
    echo "   python examples/hdmap/analyze_experiment_results.py --results-dir ${LOG_DIR}"
    echo ""
    echo ""
    echo "📈 TensorBoard 시각화:"
    echo "   tensorboard --logdir ${LOG_DIR} --port 6006"
    echo ""
    
    # 최종 통계
    echo "📈 실험 완료 통계:"
    echo "   전체 실험: ${NUM_EXPERIMENTS}개"
    echo "   성공: ${successful_experiments}개"
    echo "   실패: ${failed_experiments}개"
    
    if [ ${NUM_EXPERIMENTS} -gt 0 ]; then
        SUCCESS_RATE=$(echo "scale=1; ${successful_experiments} * 100 / ${NUM_EXPERIMENTS}" | bc)
        echo "   성공률: ${SUCCESS_RATE}%"
        
        if [ ${successful_experiments} -eq ${NUM_EXPERIMENTS} ]; then
            echo ""
            echo "🎊 모든 실험이 성공적으로 완료되었습니다!"
        elif [ ${successful_experiments} -gt 0 ]; then
            echo ""
            echo "⚠️ 일부 실험이 실패했습니다. 로그를 확인해주세요."
        else
            echo ""
            echo "❌ 모든 실험이 실패했습니다. 설정을 확인해주세요."
        fi
    else
        echo ""
        echo "⚠️ 실행할 실험이 없습니다."
    fi
else
    echo "⚠️ 실행할 실험이 없습니다."
fi

echo ""
echo "🏁 EfficientAD 병렬 실험 완료!"
