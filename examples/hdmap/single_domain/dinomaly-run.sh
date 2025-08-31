#!/bin/bash
# nohup ./examples/hdmap/single_domain/dinomaly-run.sh > /dev/null 2>&1 &
# pkill -f "single_domain/dinomaly-run.sh"
# pkill -f "examples/hdmap/single_domain/dinomaly-training.py"

# Dinomaly Single Domain HDMAP Experiments Runner
# 이 스크립트는 Dinomaly 모델의 단일 도메인 실험을 실행합니다.

set -e  # 오류 발생시 스크립트 중단

# 스크립트 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/dinomaly-training.py"

# 로그 디렉토리 생성
LOG_DIR="$PROJECT_ROOT/logs/dinomaly_single_domain"
mkdir -p "$LOG_DIR"

# 타임스탬프
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=================================================================="
echo "Dinomaly Single Domain HDMAP Experiments"
echo "=================================================================="
echo "시작 시간: $(date)"
echo "프로젝트 루트: $PROJECT_ROOT"
echo "Python 스크립트: $PYTHON_SCRIPT"
echo "로그 디렉토리: $LOG_DIR"
echo "=================================================================="

# Python 환경 활성화
echo "Python 가상환경 활성화 중..."
cd "$PROJECT_ROOT"
source .venv/bin/activate

# Python 스크립트 존재 확인
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "오류: Python 스크립트를 찾을 수 없습니다: $PYTHON_SCRIPT"
    exit 1
fi

# GPU 메모리 정리
echo "GPU 메모리 정리 중..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# 실험 실행
echo "Dinomaly 단일 도메인 실험 시작..."
LOG_FILE="$LOG_DIR/dinomaly_single_domain_$TIMESTAMP.log"

# Python 스크립트 실행 (출력을 로그 파일과 콘솔 모두에 표시)
python "$PYTHON_SCRIPT" 2>&1 | tee "$LOG_FILE"

# 실행 결과 확인
PYTHON_EXIT_CODE=${PIPESTATUS[0]}

echo "=================================================================="
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "모든 실험이 성공적으로 완료되었습니다!"
    echo "로그 파일: $LOG_FILE"
else
    echo "실험 실행 중 오류가 발생했습니다. (종료 코드: $PYTHON_EXIT_CODE)"
    echo "로그 파일을 확인하세요: $LOG_FILE"
fi

echo "종료 시간: $(date)"
echo "=================================================================="

# 결과 분석 제안
echo ""
echo "실험 완료 후 다음 명령으로 결과를 분석할 수 있습니다:"
echo "python examples/hdmap/analyze_experiment_results.py --results_dir results/dinomaly"

# 가상환경 비활성화
deactivate

exit $PYTHON_EXIT_CODE