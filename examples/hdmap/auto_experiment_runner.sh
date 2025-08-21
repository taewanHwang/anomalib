#!/bin/bash
# nohup examples/hdmap/auto_experiment_runner.sh -s examples/hdmap/multi_domain_hdmap_draem-run.sh 3 > auto_experiment_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# nohup examples/hdmap/auto_experiment_runner.sh -s examples/hdmap/multi_domain_hdmap_draem_sevnet-run.sh 10 > auto_experiment_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# GPU 모니터링 기반 자동 실험 실행 스크립트
# GPU가 모두 유휴 상태가 되면 자동으로 다음 실험을 시작합니다

set -e  # 오류 발생 시 중단

# =============================================================================
# 설정 변수
# =============================================================================

# 기본 설정
EXPERIMENT_SCRIPT=""  # 필수 옵션으로 변경
RESULTS_BASE_DIR="results/draem"
DEFAULT_EXPERIMENTS=3

# GPU 모니터링 설정
GPU_CHECK_INTERVAL=30          # GPU 상태 확인 간격 (초)
GPU_IDLE_THRESHOLD=10          # GPU 사용률 임계값 (% 이하면 유휴)
MEMORY_IDLE_THRESHOLD=2000     # 메모리 사용량 임계값 (MB 이하면 유휴)
MAX_WAIT_TIME=7200             # 최대 대기 시간 (초, 2시간)

# 로그 설정
LOG_PREFIX="auto_experiment"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_PREFIX}_${TIMESTAMP}.log"

# =============================================================================
# 함수 정의
# =============================================================================

# 로그 출력 함수
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# 도움말 출력
show_help() {
    cat << EOF
GPU 모니터링 기반 자동 실험 실행 스크립트

사용법:
    $0 -s <실험_스크립트> [OPTIONS] <실험_횟수>

필수 옵션:
    -s, --script PATH       실험 스크립트 경로 (필수)

기타 옵션:
    -r, --results PATH      결과 저장 디렉토리 (기본: $RESULTS_BASE_DIR)
    -i, --interval SEC      GPU 확인 간격 (기본: ${GPU_CHECK_INTERVAL}초)
    -t, --threshold PERCENT GPU 유휴 임계값 (기본: ${GPU_IDLE_THRESHOLD}%)
    -w, --wait SEC          최대 대기 시간 (기본: ${MAX_WAIT_TIME}초)
    -c, --check-only        GPU 상태만 확인하고 종료
    -h, --help              이 도움말 출력

예시:
    $0 -s examples/hdmap/multi_domain_hdmap_draem-run.sh 5     # DRAEM 스크립트로 5회 반복
    $0 -s examples/hdmap/multi_domain_hdmap_draem_sevnet-run.sh 5     # DRAEM-SevNet 스크립트로 5회 반복
    $0 -s my_experiment.sh -i 60 -t 5 3                                # 커스텀 스크립트, 60초 간격, 5% 임계값으로 3회
    $0 --check-only                                                     # GPU 상태만 확인
    $0 -s examples/hdmap/my_custom_experiment.sh 2                     # 커스텀 스크립트로 2회 실험

EOF
}

# GPU 상태 확인 함수
check_gpu_status() {
    local show_output=${1:-false}
    
    if ! command -v nvidia-smi &> /dev/null; then
        log "ERROR" "nvidia-smi를 찾을 수 없습니다. NVIDIA GPU가 설치되어 있는지 확인하세요."
        return 1
    fi
    
    # GPU 정보 조회
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
               --format=csv,noheader,nounits 2>/dev/null)
    
    if [[ $? -ne 0 ]] || [[ -z "$gpu_info" ]]; then
        log "ERROR" "GPU 정보 조회에 실패했습니다."
        return 1
    fi
    
    local all_idle=true
    local idle_gpus=()
    local busy_gpus=()
    
    while IFS=',' read -r index util mem_used mem_total temp; do
        # 공백 제거
        index=$(echo "$index" | tr -d ' ')
        util=$(echo "$util" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' ')
        temp=$(echo "$temp" | tr -d ' ')
        
        if [[ $show_output == true ]]; then
            log "INFO" "GPU $index: ${util}% 사용률, ${mem_used}MB 메모리, ${temp}°C"
        fi
        
        # 유휴 상태 판단
        if [[ $util -le $GPU_IDLE_THRESHOLD ]] && [[ $mem_used -le $MEMORY_IDLE_THRESHOLD ]]; then
            idle_gpus+=("$index")
        else
            busy_gpus+=("$index(${util}%,${mem_used}MB)")
            all_idle=false
        fi
        
    done <<< "$gpu_info"
    
    if [[ $show_output == true ]]; then
        if [[ $all_idle == true ]]; then
            log "INFO" "✅ 모든 GPU가 유휴 상태입니다"
            log "INFO" "   유휴 GPU: ${idle_gpus[*]}"
        else
            log "INFO" "⏳ 일부 GPU가 사용 중입니다"
            log "INFO" "   유휴 GPU: ${idle_gpus[*]}"
            log "INFO" "   사용 중 GPU: ${busy_gpus[*]}"
        fi
    fi
    
    [[ $all_idle == true ]]
}

# GPU 유휴 상태 대기 함수
wait_for_gpu_idle() {
    local start_time=$(date +%s)
    local end_time=$((start_time + MAX_WAIT_TIME))
    
    log "INFO" "🔍 GPU 유휴 상태 대기 시작 (최대 ${MAX_WAIT_TIME}초)"
    log "INFO" "   모니터링 간격: ${GPU_CHECK_INTERVAL}초"
    log "INFO" "   유휴 임계값: 사용률 ${GPU_IDLE_THRESHOLD}%, 메모리 ${MEMORY_IDLE_THRESHOLD}MB"
    
    while [[ $(date +%s) -lt $end_time ]]; do
        if check_gpu_status false; then
            log "INFO" "✅ 모든 GPU가 유휴 상태가 되었습니다!"
            return 0
        fi
        
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        local remaining=$((MAX_WAIT_TIME - elapsed))
        
        log "INFO" "⏳ 대기 중... (경과: ${elapsed}초, 남은시간: ${remaining}초)"
        check_gpu_status true
        
        sleep "$GPU_CHECK_INTERVAL"
    done
    
    log "ERROR" "⚠️ 타임아웃: ${MAX_WAIT_TIME}초 내에 GPU가 유휴 상태가 되지 않았습니다"
    return 1
}

# 실험 실행 함수
run_single_experiment() {
    local experiment_num=$1
    local total_experiments=$2
    
    log "INFO" "🚀 실험 ${experiment_num}/${total_experiments} 시작"
    log "INFO" "   스크립트: $EXPERIMENT_SCRIPT"
    
    local start_time=$(date +%s)
    
    # 실험 스크립트 실행 (subshell에서 안전하게 실행)
    log "INFO" "🔍 디버그: 실험 스크립트 실행 시작"
    (
        # subshell에서 실행하여 메인 shell에 영향 없도록 함
        bash "$EXPERIMENT_SCRIPT"
    )
    local exit_code=$?
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $exit_code -eq 0 ]]; then
        log "INFO" "✅ 실험 ${experiment_num} 완료 (소요시간: ${duration}초)"
        log "INFO" "🔍 디버그: run_single_experiment 함수 정상 반환"
        return 0
    else
        log "ERROR" "❌ 실험 ${experiment_num} 실패 (소요시간: ${duration}초, exit_code: $exit_code)"
        log "INFO" "🔍 디버그: run_single_experiment 함수 오류 반환"
        return 1
    fi
}

# 시그널 핸들러
cleanup() {
    log "INFO" "🛑 사용자 중단 신호를 받았습니다. 정리 작업을 수행합니다..."
    
    # 자식 프로세스들 종료
    local children=$(jobs -p)
    if [[ -n "$children" ]]; then
        log "INFO" "자식 프로세스들을 종료합니다: $children"
        kill $children 2>/dev/null
        wait $children 2>/dev/null
    fi
    
    log "INFO" "프로그램을 종료합니다."
    exit 130
}

# =============================================================================
# 메인 로직
# =============================================================================

# 시그널 핸들러 등록
trap cleanup SIGINT SIGTERM

# 인자 파싱
CHECK_ONLY=false
NUM_EXPERIMENTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--script)
            EXPERIMENT_SCRIPT="$2"
            shift 2
            ;;
        -r|--results)
            RESULTS_BASE_DIR="$2"
            shift 2
            ;;
        -i|--interval)
            GPU_CHECK_INTERVAL="$2"
            shift 2
            ;;
        -t|--threshold)
            GPU_IDLE_THRESHOLD="$2"
            shift 2
            ;;
        -w|--wait)
            MAX_WAIT_TIME="$2"
            shift 2
            ;;
        -c|--check-only)
            CHECK_ONLY=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            log "ERROR" "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
        *)
            if [[ -z "$NUM_EXPERIMENTS" ]]; then
                NUM_EXPERIMENTS="$1"
            else
                log "ERROR" "잘못된 인자: $1"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# GPU 상태만 확인하고 종료
if [[ $CHECK_ONLY == true ]]; then
    log "INFO" "🔍 GPU 상태 확인 모드"
    echo
    check_gpu_status true
    exit 0
fi

# 실험 스크립트 검증 (필수)
if [[ -z "$EXPERIMENT_SCRIPT" ]]; then
    log "ERROR" "실험 스크립트를 지정해주세요 (-s 옵션 필수)"
    echo ""
    show_help
    exit 1
fi

# 실험 스크립트 존재 확인
if [[ ! -f "$EXPERIMENT_SCRIPT" ]]; then
    log "ERROR" "실험 스크립트를 찾을 수 없습니다: $EXPERIMENT_SCRIPT"
    exit 1
fi

# 실험 횟수 검증
if [[ -z "$NUM_EXPERIMENTS" ]]; then
    log "ERROR" "실험 횟수를 지정해주세요"
    show_help
    exit 1
fi

if ! [[ "$NUM_EXPERIMENTS" =~ ^[0-9]+$ ]] || [[ $NUM_EXPERIMENTS -le 0 ]]; then
    log "ERROR" "실험 횟수는 양의 정수여야 합니다: $NUM_EXPERIMENTS"
    exit 1
fi

# 결과 디렉토리 생성
mkdir -p "$RESULTS_BASE_DIR"

# 시작 정보 출력
log "INFO" "=" | tr ' ' '='  # 구분선
log "INFO" "🎯 자동 실험 실행 시작"
log "INFO" "=" | tr ' ' '='  # 구분선
log "INFO" "📊 설정 정보:"
log "INFO" "   총 실험 횟수: $NUM_EXPERIMENTS"
log "INFO" "   실험 스크립트: $EXPERIMENT_SCRIPT"
log "INFO" "   결과 디렉토리: $RESULTS_BASE_DIR"
log "INFO" "   GPU 확인 간격: ${GPU_CHECK_INTERVAL}초"
log "INFO" "   GPU 유휴 임계값: ${GPU_IDLE_THRESHOLD}%"
log "INFO" "   메모리 임계값: ${MEMORY_IDLE_THRESHOLD}MB"
log "INFO" "   최대 대기 시간: ${MAX_WAIT_TIME}초"
log "INFO" "   로그 파일: $LOG_FILE"

# 초기 GPU 상태 확인
log "INFO" "🔍 초기 GPU 상태 확인:"
check_gpu_status true

# 실험 실행
successful_experiments=0
failed_experiments=0

for ((i=1; i<=NUM_EXPERIMENTS; i++)); do
    log "INFO" ""
    log "INFO" "=========================================="
    log "INFO" "📋 실험 $i/$NUM_EXPERIMENTS 준비"
    log "INFO" "=========================================="
    log "INFO" "🔍 디버그: 반복문 시작 - 현재 i=$i, NUM_EXPERIMENTS=$NUM_EXPERIMENTS"
    
    # 첫 번째 실험이 아니면 GPU 유휴 상태 대기
    if [[ $i -gt 1 ]]; then
        log "INFO" "⏳ 이전 실험 완료 대기 중..."
        
        if ! wait_for_gpu_idle; then
            log "ERROR" "❌ GPU 대기 타임아웃. 실험 $i 건너뜀"
            ((failed_experiments++))
            continue
        fi
        
        # 추가 안전 대기 시간
        log "INFO" "😴 안전을 위해 30초 추가 대기..."
        sleep 30
    fi
    
    # 실험 실행
    log "INFO" "🔍 디버그: run_single_experiment 호출 전"
    if run_single_experiment "$i" "$NUM_EXPERIMENTS"; then
        log "INFO" "🔍 디버그: run_single_experiment 성공 반환"
        log "INFO" "🔍 디버그: successful_experiments 증가 전: $successful_experiments"
        successful_experiments=$((successful_experiments + 1))
        log "INFO" "🔍 디버그: successful_experiments 증가 후: $successful_experiments"
        log "INFO" "🎉 실험 $i 성공!"
    else
        log "INFO" "🔍 디버그: run_single_experiment 실패 반환"
        log "INFO" "🔍 디버그: failed_experiments 증가 전: $failed_experiments"
        failed_experiments=$((failed_experiments + 1))
        log "INFO" "🔍 디버그: failed_experiments 증가 후: $failed_experiments"
        log "ERROR" "💥 실험 $i 실패!"
    fi
    log "INFO" "🔍 디버그: 실험 실행 블록 완료"
    
    # 중간 상태 출력
    total_completed=$((successful_experiments + failed_experiments))
    success_rate=0
    if [[ $total_completed -gt 0 ]]; then
        success_rate=$((successful_experiments * 100 / total_completed))
    fi
    
    log "INFO" "📊 현재 상태: 성공 $successful_experiments, 실패 $failed_experiments (성공률: ${success_rate}%)"
    log "INFO" "🔄 실험 $i 완료. 다음 실험 준비 중..."
done

# 최종 결과 출력
total_experiments=$((successful_experiments + failed_experiments))
final_success_rate=0
if [[ $total_experiments -gt 0 ]]; then
    final_success_rate=$((successful_experiments * 100 / total_experiments))
fi

log "INFO" ""
log "INFO" "=============================================="
log "INFO" "🏁 모든 실험 완료!"
log "INFO" "=============================================="
log "INFO" "📊 최종 결과:"
log "INFO" "   전체 실험: $total_experiments개"
log "INFO" "   성공: $successful_experiments개"
log "INFO" "   실패: $failed_experiments개"
log "INFO" "   성공률: ${final_success_rate}%"
log "INFO" "📄 상세 로그: $LOG_FILE"

# 종료 코드 설정
if [[ $failed_experiments -eq 0 ]]; then
    log "INFO" "✅ 모든 실험이 성공적으로 완료되었습니다!"
    exit 0
else
    log "WARN" "⚠️ 일부 실험이 실패했습니다."
    exit 1
fi
