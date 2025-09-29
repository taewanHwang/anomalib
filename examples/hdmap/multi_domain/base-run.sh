#!/bin/bash

# Multi-Domain Anomaly Detection 통합 실험 병렬 실행 스크립트
# 
# 🚀 기본 사용법:
#   ./examples/hdmap/multi_domain/base-run.sh               # 사용법 안내
#   ./examples/hdmap/multi_domain/base-run.sh 0             # 특정 실험 (ID 0)
#   ./examples/hdmap/multi_domain/base-run.sh 0,1,2         # 여러 실험 (ID 0,1,2)
#   ./examples/hdmap/multi_domain/base-run.sh all           # 전체 실험 (멀티 GPU 자동 할당)
#
# 🔥 백그라운드 실행 (추천):
#   nohup ./examples/hdmap/multi_domain/base-run.sh all > multi_domain_training.log 2>&1 &
#   nohup ./examples/hdmap/multi_domain/base-run.sh 2 > patchcore_test.log 2>&1 &
#
# 📊 실행 상태 확인:
#   tail -f multi_domain_training.log                      # 메인 스크립트 로그 확인  
#   tail -f results/*/multi_domain_*.log                   # 개별 실험 상세 로그 확인
#   tail -f results/*/training_detail.log                  # 실험별 훈련 상세 로그 확인
#   ps aux | grep base-run.sh                              # 스크립트 실행 여부
#   ps aux | grep base-training                            # 개별 실험 진행 상황
#   nvidia-smi                                             # GPU 사용 현황
#
# 🛑 실행 중단:
#   pkill -f "multi_domain.*base-run.sh"                   # 메인 스크립트 종료
#   pkill -f "multi_domain.*base-training.py"              # 모든 multi-domain 실험 종료
# 
# 🖥️ GPU 설정: AVAILABLE_GPUS 배열을 수정하세요
# 
# 📋 Multi-Domain 특징:
#   - Source domain(A)에서 훈련 → Target domains(B,C,D)에서 평가
#   - Transfer learning 성능 측정 (도메인 간 일반화)
#   - 결과: source AUROC + target별 AUROC + transfer ratio

set -e  # 오류 시 즉시 종료

# =============================================================================
# 설정 변수
# =============================================================================

# 사용할 GPU 목록 (0부터 시작, 사용 가능한 GPU ID를 나열)
AVAILABLE_GPUS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

# 실험 설정 파일 및 실행 스크립트 경로
SCRIPT_PATH="examples/hdmap/multi_domain/base-training.py"
CONFIG_PATH="examples/hdmap/multi_domain/base-exp_condition_quick_test.json"

# 세션 타임스탬프 (모든 실험에서 공유)
SESSION_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 로그 레벨
LOG_LEVEL="INFO"

# =============================================================================
# 유틸리티 함수들
# =============================================================================

print_header() {
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
}

print_section() {
    echo ""
    echo "🔹 $1"
    echo "--------------------------------------------------------------------------------"
}

check_prerequisites() {
    print_section "전제 조건 확인 중..."
    
    # 파이썬 가상환경 확인
    if [[ -z "$VIRTUAL_ENV" && ! -f ".venv/bin/activate" ]]; then
        echo "❌ 오류: Python 가상환경이 활성화되지 않았습니다."
        echo "다음 명령어로 가상환경을 활성화하세요:"
        echo "  source .venv/bin/activate"
        exit 1
    fi
    
    # 설정 파일 존재 확인
    if [[ ! -f "$CONFIG_PATH" ]]; then
        echo "❌ 오류: 설정 파일을 찾을 수 없습니다: $CONFIG_PATH"
        exit 1
    fi
    
    # 실행 스크립트 존재 확인
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "❌ 오류: 실행 스크립트를 찾을 수 없습니다: $SCRIPT_PATH"
        exit 1
    fi
    
    # GPU 사용 가능 확인
    if ! command -v nvidia-smi &> /dev/null; then
        echo "⚠️ 경고: nvidia-smi를 찾을 수 없습니다. GPU가 사용 가능한지 확인하세요."
    else
        echo "✅ GPU 상태 확인:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | head -4
    fi
    
    echo "✅ 전제 조건 확인 완료"
}

get_total_experiments() {
    # JSON 파일에서 실험 조건 개수 확인
    local count=$(python3 -c "
import json
with open('$CONFIG_PATH', 'r') as f:
    data = json.load(f)
print(len(data.get('experiment_conditions', [])))
")
    echo "$count"
}

validate_experiment_ids() {
    local experiment_ids="$1"
    local total_experiments=$(get_total_experiments)
    
    # "all" 처리
    if [[ "$experiment_ids" == "all" ]]; then
        # 0부터 (total_experiments-1)까지의 배열 생성
        EXPERIMENT_IDS=($(seq 0 $((total_experiments-1))))
        return
    fi
    
    # 콤마로 구분된 ID들 처리
    IFS=',' read -ra ID_ARRAY <<< "$experiment_ids"
    EXPERIMENT_IDS=()
    
    for id in "${ID_ARRAY[@]}"; do
        # 공백 제거
        id=$(echo "$id" | tr -d ' ')
        
        # 숫자인지 확인
        if ! [[ "$id" =~ ^[0-9]+$ ]]; then
            echo "❌ 오류: 유효하지 않은 실험 ID: $id"
            exit 1
        fi
        
        # 범위 확인
        if [[ "$id" -ge "$total_experiments" ]]; then
            echo "❌ 오류: 실험 ID $id가 범위를 벗어났습니다. (최대: $((total_experiments-1)))"
            exit 1
        fi
        
        EXPERIMENT_IDS+=("$id")
    done
}

get_experiment_info() {
    local experiment_id=$1
    python3 -c "
import json
with open('$CONFIG_PATH', 'r') as f:
    data = json.load(f)
conditions = data.get('experiment_conditions', [])
if $experiment_id < len(conditions):
    condition = conditions[$experiment_id]
    print(f\"{condition['name']}:{condition['config']['model_type']}\")
else:
    print(\"unknown:unknown\")
"
}

wait_for_gpu() {
    local gpu_id=$1
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        # GPU 사용률 확인 (메모리 기준)
        local gpu_memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id 2>/dev/null || echo "0")
        
        # GPU 메모리 사용량이 1GB 미만이면 사용 가능으로 판단
        if [[ "$gpu_memory_used" -lt 1024 ]]; then
            return 0
        fi
        
        echo "   ⏳ GPU $gpu_id 대기 중... (시도 $((attempt+1))/$max_attempts, 메모리 사용량: ${gpu_memory_used}MB)"
        sleep 10
        ((attempt++))
    done
    
    echo "   ⚠️ 경고: GPU $gpu_id가 계속 사용 중입니다. 강제로 실행합니다."
    return 0
}

cleanup() {
    print_section "정리 작업 수행 중..."
    
    # 백그라운드 작업들 확인
    local running_jobs=$(jobs -r | wc -l)
    if [[ $running_jobs -gt 0 ]]; then
        echo "실행 중인 작업 $running_jobs개를 정리합니다..."
        jobs -p | xargs -r kill -TERM 2>/dev/null || true
        sleep 5
        jobs -p | xargs -r kill -KILL 2>/dev/null || true
    fi
    
    echo "✅ 정리 작업 완료"
}

# =============================================================================
# 메인 실행 로직
# =============================================================================

run_experiments() {
    local experiment_ids_input="$1"
    
    print_header "🚀 Multi-Domain Anomaly Detection 병렬 실험 시작"
    
    echo "📅 실험 세션: $SESSION_TIMESTAMP"
    echo "💻 사용 가능한 GPU: ${AVAILABLE_GPUS[*]}"
    echo "📋 설정 파일: $CONFIG_PATH"
    echo "🔧 로그 레벨: $LOG_LEVEL"
    
    # 실험 ID 검증 및 준비
    validate_experiment_ids "$experiment_ids_input"
    local total_experiments=${#EXPERIMENT_IDS[@]}
    local total_available_gpus=${#AVAILABLE_GPUS[@]}
    
    echo "🧪 실행할 실험 수: $total_experiments"
    echo "🎯 실험 ID 목록: ${EXPERIMENT_IDS[*]}"
    
    # 실험 정보 출력
    print_section "실험 정보 미리보기"
    for exp_id in "${EXPERIMENT_IDS[@]}"; do
        local exp_info=$(get_experiment_info $exp_id)
        local exp_name=$(echo "$exp_info" | cut -d: -f1)
        local model_type=$(echo "$exp_info" | cut -d: -f2)
        echo "  ID $exp_id: $exp_name ($model_type)"
    done
    
    # 실험 실행
    print_section "실험 병렬 실행 시작"
    
    local gpu_index=0
    local running_experiments=0
    local completed_experiments=0
    local failed_experiments=0
    
    # 각 실험을 GPU에 할당하여 실행
    for exp_id in "${EXPERIMENT_IDS[@]}"; do
        local gpu_id=${AVAILABLE_GPUS[$gpu_index]}
        local exp_info=$(get_experiment_info $exp_id)
        local exp_name=$(echo "$exp_info" | cut -d: -f1)
        local model_type=$(echo "$exp_info" | cut -d: -f2)
        
        echo ""
        echo "🔄 실험 $exp_id 시작 준비..."
        echo "   📝 이름: $exp_name"
        echo "   🤖 모델: $model_type" 
        echo "   💻 GPU: $gpu_id"
        
        # GPU 사용 가능할 때까지 대기
        wait_for_gpu $gpu_id
        
        # 실험 실행 (백그라운드)
        echo "   🚀 실험 실행 시작..."
        
        # 실험 디렉터리 생성 (single domain과 동일)
        local experiment_dir="results/$SESSION_TIMESTAMP/${exp_name}_${SESSION_TIMESTAMP}"
        mkdir -p "$experiment_dir"
        local training_log="$experiment_dir/training_detail.log"
        
        (
            # 가상환경이 활성화되어 있지 않으면 활성화
            if [[ -z "$VIRTUAL_ENV" && -f ".venv/bin/activate" ]]; then
                source .venv/bin/activate
            fi
            
            echo "🔬 [Exp$exp_id-GPU$gpu_id] 실험 시작" >> "$training_log"
            
            # Python 스크립트를 실험 디렉터리의 로그 파일로 실행 (single domain과 동일)
            python "$SCRIPT_PATH" \
                --config "$CONFIG_PATH" \
                --experiment-id $exp_id \
                --gpu-id $gpu_id \
                --session-timestamp "$SESSION_TIMESTAMP" \
                --log-level "$LOG_LEVEL" \
                >> "$training_log" 2>&1
            
            echo "✅ [Exp$exp_id-GPU$gpu_id] 실험 완료" >> "$training_log"
        ) &
        
        local job_pid=$!
        echo "   🔢 프로세스 ID: $job_pid"
        
        ((running_experiments++))
        
        # 다음 GPU로 순환
        gpu_index=$(( (gpu_index + 1) % total_available_gpus ))
        
        # 모든 GPU가 사용 중이면 하나 완료될 때까지 대기
        if [[ $running_experiments -ge $total_available_gpus ]]; then
            echo "   ⏳ GPU가 모두 사용 중입니다. 실험 완료 대기..."
            wait -n  # 아무 작업이나 완료될 때까지 대기
            ((running_experiments--))
            ((completed_experiments++))
        fi
        
        sleep 2  # GPU 초기화 시간 확보
    done
    
    # 남은 실험들 완료 대기
    print_section "남은 실험들 완료 대기 중..."
    echo "대기 중인 실험 수: $running_experiments"
    
    while [[ $running_experiments -gt 0 ]]; do
        wait -n
        ((running_experiments--))
        ((completed_experiments++))
        echo "완료된 실험 수: $completed_experiments / $total_experiments"
    done
    
    # 최종 결과 확인
    print_section "실험 결과 확인 중..."
    
    local results_dir="results/$SESSION_TIMESTAMP"
    if [[ -d "$results_dir" ]]; then
        local success_count=$(find "$results_dir" -name "result_*.json" | wc -l)
        local log_count=$(find "$results_dir" -name "multi_domain_*.log" | wc -l)
        
        echo "📊 결과 요약:"
        echo "   📁 결과 디렉터리: $results_dir"
        echo "   ✅ 성공한 실험: $success_count"
        echo "   📝 로그 파일: $log_count"
        
        if [[ $success_count -eq $total_experiments ]]; then
            echo "🎉 모든 실험이 성공적으로 완료되었습니다!"
        else
            echo "⚠️ 일부 실험이 실패했을 수 있습니다."
            failed_experiments=$((total_experiments - success_count))
        fi
    else
        echo "❌ 결과 디렉터리를 찾을 수 없습니다: $results_dir"
        failed_experiments=$total_experiments
    fi
    
    print_header "🏁 Multi-Domain 병렬 실험 완료"
    echo "📅 완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "📊 실행한 실험 수: $total_experiments"
    echo "✅ 성공한 실험: $((total_experiments - failed_experiments))"
    echo "❌ 실패한 실험: $failed_experiments"
    echo "📁 결과 경로: $results_dir"
    echo ""
    echo "📋 결과 분석 명령어:"
    echo "  source .venv/bin/activate && python examples/hdmap/analyze_experiment_results.py --results_dir results --all-models"
    echo ""
    echo "📝 로그 확인 명령어:"
    echo "  tail -f $results_dir/*/multi_domain_*.log"
    echo "  grep -r 'AUROC' $results_dir/"
    
    return $failed_experiments
}

# =============================================================================
# 스크립트 진입점
# =============================================================================

main() {
    # 신호 핸들러 설정
    trap cleanup EXIT INT TERM
    
    # 인수 확인
    if [[ $# -eq 0 ]]; then
        echo "사용법: $0 <실험_ID>"
        echo ""
        echo "옵션:"
        echo "  all           모든 실험 실행"
        echo "  0             실험 ID 0만 실행"
        echo "  0,1,2         실험 ID 0, 1, 2 실행"
        echo ""
        echo "예시:"
        echo "  $0 all"
        echo "  $0 0"
        echo "  $0 0,5,10"
        exit 1
    fi
    
    local experiment_ids="$1"
    
    # 전제 조건 확인
    check_prerequisites
    
    # 실험 실행
    if run_experiments "$experiment_ids"; then
        echo "🎉 모든 Multi-Domain 실험이 성공적으로 완료되었습니다!"
        exit 0
    else
        echo "❌ 일부 Multi-Domain 실험이 실패했습니다."
        exit 1
    fi
}

# 스크립트가 직접 실행될 때만 main 함수 호출
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi