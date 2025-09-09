#!/bin/bash
# 
# 🚀 GPU 모니터링 기반 자동 실험 반복 실행 스크립트 (v2.0)
#
# 사용 예시:
#   nohup examples/hdmap/auto_experiment_runner.sh -s examples/hdmap/single_domain/base-run.sh -a all 3 > auto_experiment_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   nohup examples/hdmap/auto_experiment_runner.sh -s examples/hdmap/single_domain/base-run.sh -a 0,1,2 5 > auto_experiment_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# 주요 변경사항:
#   - single_domain/base-run.sh 새로운 구조 지원
#   - 실험 인자(-a) 옵션 추가 (all, 특정 ID, ID 범위 등)
#   - 더 정확한 GPU 모니터링
#   - 실험별 결과 디렉터리 자동 관리

set -e  # 오류 발생 시 중단

# =============================================================================
# 설정 변수
# =============================================================================

# 기본 설정
EXPERIMENT_SCRIPT=""              # 필수: 실험 스크립트 경로
EXPERIMENT_ARGS="all"             # 실험 인자 (all, 0, 0,1,2 등)
DEFAULT_EXPERIMENTS=3

# GPU 모니터링 설정
GPU_CHECK_INTERVAL=30             # GPU 상태 확인 간격 (초)
GPU_IDLE_THRESHOLD=10             # GPU 사용률 임계값 (% 이하면 유휴)
MEMORY_IDLE_THRESHOLD=2000        # 메모리 사용량 임계값 (MB 이하면 유휴)
MAX_WAIT_TIME=7200                # 최대 대기 시간 (초, 2시간)
SAFETY_WAIT=60                    # 실험 사이 안전 대기 시간 (초)

# 로그 설정
LOG_PREFIX="auto_experiment"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_PREFIX}_${TIMESTAMP}.log"

# 결과 디렉터리 설정 (base-run.sh가 자체적으로 results/timestamp 폴더 생성)
RESULTS_BASE_DIR="results"

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
🚀 GPU 모니터링 기반 자동 실험 반복 실행 스크립트 (v2.0)

사용법:
    $0 -s <실험_스크립트> [-a <실험_인자>] [OPTIONS] <반복_횟수>

필수 옵션:
    -s, --script PATH           실험 스크립트 경로 (필수)

선택 옵션:
    -a, --args ARGS             실험 인자 (기본: all)
                                - all: 전체 실험
                                - 0: 특정 실험 ID
                                - 0,1,2: 여러 실험 ID
                                - 0-5: 실험 ID 범위
    -r, --results PATH          결과 저장 기본 디렉토리 (기본: $RESULTS_BASE_DIR)
    -i, --interval SEC          GPU 확인 간격 (기본: ${GPU_CHECK_INTERVAL}초)
    -t, --threshold PERCENT     GPU 유휴 임계값 (기본: ${GPU_IDLE_THRESHOLD}%)
    -w, --wait SEC              최대 대기 시간 (기본: ${MAX_WAIT_TIME}초)
    -safety, --safety-wait SEC  실험 사이 안전 대기 시간 (기본: ${SAFETY_WAIT}초)
    -c, --check-only            GPU 상태만 확인하고 종료
    -h, --help                  이 도움말 출력

예시:
    # 전체 실험을 3회 반복 (가장 일반적)
    $0 -s examples/hdmap/single_domain/base-run.sh -a all 3

    # 특정 실험(0,1,2)을 5회 반복
    $0 -s examples/hdmap/single_domain/base-run.sh -a 0,1,2 5

    # 커스텀 설정으로 실험
    $0 -s examples/hdmap/single_domain/base-run.sh -a all -i 60 -t 5 --safety-wait 120 3

    # DinomaLy 실험만 2회 반복 (condition1.json 사용)
    $0 -s examples/hdmap/single_domain/base-run.sh -a 0 2

    # GPU 상태만 확인
    $0 --check-only

백그라운드 실행 (추천):
    nohup $0 -s examples/hdmap/single_domain/base-run.sh -a all 3 > auto_experiment_\$(date +%Y%m%d_%H%M%S).log 2>&1 &

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
        
        # 유휴 상태 판단 (더 엄격한 기준)
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

# 실험 결과 디렉터리 정리 함수
cleanup_old_results() {
    log "INFO" "🧹 이전 실험 결과 정리 중..."
    
    # 3시간 이상 된 임시 파일들 정리
    find "$RESULTS_BASE_DIR" -type f -name "*.tmp" -mtime +0.125 -delete 2>/dev/null || true
    
    # GPU 프로세스가 완전히 종료되었는지 확인
    if pgrep -f "base-training.py" > /dev/null; then
        log "WARN" "⚠️ 이전 training 프로세스가 아직 실행 중입니다"
        log "INFO" "   대기 중인 프로세스:"
        pgrep -af "base-training.py" || true
    fi
}

# 단일 실험 실행 함수
run_single_experiment() {
    local experiment_num=$1
    local total_experiments=$2
    
    log "INFO" "🚀 실험 ${experiment_num}/${total_experiments} 시작"
    log "INFO" "   스크립트: $EXPERIMENT_SCRIPT"
    log "INFO" "   인자: $EXPERIMENT_ARGS"
    
    local start_time=$(date +%s)
    
    # 실험 스크립트를 직접 실행 (base-run.sh가 자체적으로 results/timestamp 폴더 생성)
    local exit_code=0
    bash "$EXPERIMENT_SCRIPT" $EXPERIMENT_ARGS
    exit_code=$?
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $exit_code -eq 0 ]]; then
        log "INFO" "✅ 실험 ${experiment_num} 완료 (소요시간: ${duration}초)"
        return 0
    else
        log "ERROR" "❌ 실험 ${experiment_num} 실패 (소요시간: ${duration}초, exit_code: $exit_code)"
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
    
    # 실행 중인 training 프로세스들도 종료
    if pgrep -f "base-training.py" > /dev/null; then
        log "INFO" "실행 중인 training 프로세스들을 종료합니다..."
        pkill -f "base-training.py" 2>/dev/null || true
        sleep 5
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
        -a|--args)
            EXPERIMENT_ARGS="$2"
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
        -safety|--safety-wait)
            SAFETY_WAIT="$2"
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
    log "ERROR" "실험 반복 횟수를 지정해주세요"
    show_help
    exit 1
fi

if ! [[ "$NUM_EXPERIMENTS" =~ ^[0-9]+$ ]] || [[ $NUM_EXPERIMENTS -le 0 ]]; then
    log "ERROR" "실험 반복 횟수는 양의 정수여야 합니다: $NUM_EXPERIMENTS"
    exit 1
fi

# 결과 디렉터리 생성 
mkdir -p "$RESULTS_BASE_DIR"

# 시작 정보 출력
log "INFO" "==============================================="
log "INFO" "🎯 자동 실험 반복 실행 시작"
log "INFO" "==============================================="
log "INFO" "📊 설정 정보:"
log "INFO" "   실험 스크립트: $EXPERIMENT_SCRIPT"
log "INFO" "   실험 인자: $EXPERIMENT_ARGS"
log "INFO" "   반복 횟수: $NUM_EXPERIMENTS"
log "INFO" "   결과 디렉토리: $RESULTS_BASE_DIR (base-run.sh가 타임스탬프 폴더 생성)"
log "INFO" "   GPU 확인 간격: ${GPU_CHECK_INTERVAL}초"
log "INFO" "   GPU 유휴 임계값: ${GPU_IDLE_THRESHOLD}%"
log "INFO" "   메모리 임계값: ${MEMORY_IDLE_THRESHOLD}MB"
log "INFO" "   최대 대기 시간: ${MAX_WAIT_TIME}초"
log "INFO" "   안전 대기 시간: ${SAFETY_WAIT}초"
log "INFO" "   로그 파일: $LOG_FILE"

# 초기 GPU 상태 확인
log "INFO" ""
log "INFO" "🔍 초기 GPU 상태 확인:"
check_gpu_status true

# 이전 결과 정리
cleanup_old_results

# 실험 반복 실행
successful_experiments=0
failed_experiments=0

for ((i=1; i<=NUM_EXPERIMENTS; i++)); do
    log "INFO" ""
    log "INFO" "=========================================="
    log "INFO" "📋 실험 반복 $i/$NUM_EXPERIMENTS 준비"
    log "INFO" "=========================================="
    
    # 첫 번째 실험이 아니면 GPU 유휴 상태 대기
    if [[ $i -gt 1 ]]; then
        log "INFO" "⏳ 이전 실험 완료 대기 중..."
        
        if ! wait_for_gpu_idle; then
            log "ERROR" "❌ GPU 대기 타임아웃. 실험 반복 $i 건너뜀"
            ((failed_experiments++))
            continue
        fi
        
        # 안전 대기 시간
        log "INFO" "😴 안전을 위해 ${SAFETY_WAIT}초 추가 대기..."
        sleep "$SAFETY_WAIT"
    fi
    
    # 실험 실행
    if run_single_experiment "$i" "$NUM_EXPERIMENTS"; then
        successful_experiments=$((successful_experiments + 1))
        log "INFO" "🎉 실험 반복 $i 성공!"
    else
        failed_experiments=$((failed_experiments + 1))
        log "ERROR" "💥 실험 반복 $i 실패!"
    fi
    
    # 중간 상태 출력
    total_completed=$((successful_experiments + failed_experiments))
    success_rate=0
    if [[ $total_completed -gt 0 ]]; then
        success_rate=$((successful_experiments * 100 / total_completed))
    fi
    
    log "INFO" "📊 현재 상태: 성공 $successful_experiments, 실패 $failed_experiments (성공률: ${success_rate}%)"
    log "INFO" "🔄 실험 반복 $i 완료. 다음 실험 준비 중..."
done

# 최종 결과 출력
total_experiments=$((successful_experiments + failed_experiments))
final_success_rate=0
if [[ $total_experiments -gt 0 ]]; then
    final_success_rate=$((successful_experiments * 100 / total_experiments))
fi

log "INFO" ""
log "INFO" "=============================================="
log "INFO" "🏁 모든 실험 반복 완료!"
log "INFO" "=============================================="
log "INFO" "📊 최종 결과:"
log "INFO" "   전체 실험 반복: $total_experiments회"
log "INFO" "   성공: $successful_experiments회"
log "INFO" "   실패: $failed_experiments회"
log "INFO" "   성공률: ${final_success_rate}%"
log "INFO" "📁 모든 결과: $RESULTS_BASE_DIR"
log "INFO" "📄 상세 로그: $LOG_FILE"

# 결과 요약 생성
SUMMARY_FILE="$RESULTS_BASE_DIR/experiment_summary_${TIMESTAMP}.txt"
cat > "$SUMMARY_FILE" << EOF
🎯 자동 실험 반복 실행 요약
========================================

실행 시간: $(date)
실험 스크립트: $EXPERIMENT_SCRIPT
실험 인자: $EXPERIMENT_ARGS
반복 횟수: $NUM_EXPERIMENTS

📊 결과:
- 성공: $successful_experiments회
- 실패: $failed_experiments회  
- 성공률: ${final_success_rate}%

📁 결과 위치: $RESULTS_BASE_DIR
📄 상세 로그: $LOG_FILE

========================================
EOF

log "INFO" "📋 실험 요약 파일: $SUMMARY_FILE"

# 종료 코드 설정
if [[ $failed_experiments -eq 0 ]]; then
    log "INFO" "✅ 모든 실험 반복이 성공적으로 완료되었습니다!"
    exit 0
else
    log "WARN" "⚠️ 일부 실험 반복이 실패했습니다."
    exit 1
fi