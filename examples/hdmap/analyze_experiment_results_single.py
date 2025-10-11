#!/usr/bin/env python3
"""
Single Domain 실험 결과 분석 스크립트

이 스크립트는 Single-Domain 실험의 metrics_report.json 파일을 분석하여
성능 메트릭을 테이블 형태로 출력합니다.

==============================================================================
🚀 사용법:
==============================================================================

.venv/bin/python examples/hdmap/analyze_experiment_results_single.py --results_dir results_100k_train

==============================================================================
📁 대상 폴더 구조:
==============================================================================

results_100k_train/
└── 20251011_054347/
    ├── exp-71.A_20251011_054347/
    │   └── analysis/
    │       └── metrics_report.json
    ├── exp-71.B_20251011_054347/
    │   └── analysis/
    │       └── metrics_report.json
    └── ...

==============================================================================
📊 출력 내용:
==============================================================================

실험별 상세 성능 테이블:
- 실험 이름 (타임스탬프 포함)
- 도메인 (A, B, C, D)
- AUROC
- Optimal Threshold
- Precision
- Recall
- F1 Score
- Accuracy (Confusion Matrix 기반 계산)

==============================================================================
🔧 Metrics JSON 형식:
==============================================================================

{
  "auroc": 1.0,
  "optimal_threshold": 0.41473591327667236,
  "precision": 1.0,
  "recall": 0.9988888888888889,
  "f1_score": 0.9994441356309061,
  "confusion_matrix": [[900, 0], [1, 899]],
  "total_samples": 1800,
  "positive_samples": 900,
  "negative_samples": 900
}
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def extract_experiment_info(exp_dir_name):
    """실험 디렉토리 이름에서 정보 추출

    예: exp-71.A_20251011_054347
    -> experiment_name: exp-71.A
    -> domain: A
    -> timestamp: 20251011_054347
    """
    # 타임스탬프 추출 (마지막 두 부분: YYYYMMDD_HHMMSS)
    parts = exp_dir_name.split('_')

    timestamp = None
    exp_name_with_domain = exp_dir_name

    # 마지막 두 부분이 timestamp인지 확인
    if len(parts) >= 3:
        if (parts[-2].isdigit() and len(parts[-2]) == 8 and
            parts[-1].isdigit() and len(parts[-1]) == 6):
            timestamp = f"{parts[-2]}_{parts[-1]}"
            exp_name_with_domain = '_'.join(parts[:-2])

    # 도메인 추출 (exp-71.A -> A)
    domain = None
    experiment_name = exp_name_with_domain

    if '.' in exp_name_with_domain:
        exp_parts = exp_name_with_domain.split('.')
        if len(exp_parts) >= 2:
            domain = exp_parts[1]  # A, B, C, D 등
            # experiment_name은 도메인 포함한 전체 이름
            experiment_name = exp_name_with_domain

    return {
        'full_name': exp_dir_name,
        'experiment_name': experiment_name,
        'domain': domain,
        'timestamp': timestamp
    }


def calculate_accuracy(confusion_matrix):
    """Confusion Matrix로부터 Accuracy 계산

    confusion_matrix: [[TN, FP], [FN, TP]]
    """
    if confusion_matrix and len(confusion_matrix) == 2 and len(confusion_matrix[0]) == 2:
        tn, fp = confusion_matrix[0][0], confusion_matrix[0][1]
        fn, tp = confusion_matrix[1][0], confusion_matrix[1][1]
        total = tp + tn + fp + fn
        if total > 0:
            return (tp + tn) / total
    return None


def analyze_single_domain_experiments(results_dir: str):
    """Single-Domain 실험 결과 분석"""
    results_path = Path(results_dir)

    print(f"🔍 Single-Domain 실험 결과 분석 시작...")
    print(f"📁 기본 디렉토리: {results_path}")

    # timestamp 디렉토리 찾기
    timestamp_dirs = [d for d in results_path.iterdir() if d.is_dir()]

    if not timestamp_dirs:
        print(f"❌ {results_path}에서 디렉토리를 찾을 수 없습니다.")
        return

    # 모든 실험 디렉토리 수집
    all_experiment_dirs = []
    for timestamp_dir in timestamp_dirs:
        experiment_dirs = [d for d in timestamp_dir.iterdir() if d.is_dir()]
        all_experiment_dirs.extend(experiment_dirs)

    if not all_experiment_dirs:
        print(f"❌ 실험 디렉토리를 찾을 수 없습니다.")
        return

    print(f"📊 발견된 실험 디렉토리: {len(all_experiment_dirs)}개")

    # 각 실험의 metrics_report.json 수집
    experiment_results = []

    for exp_dir in all_experiment_dirs:
        metrics_report_path = exp_dir / "analysis" / "metrics_report.json"

        if not metrics_report_path.exists():
            continue

        try:
            with open(metrics_report_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)

            # 실험 정보 추출
            exp_info = extract_experiment_info(exp_dir.name)

            # Accuracy 계산
            accuracy = calculate_accuracy(metrics_data.get('confusion_matrix'))

            # 결과 저장
            experiment_results.append({
                'experiment_name': exp_info['experiment_name'],
                'domain': exp_info['domain'],
                'timestamp': exp_info['timestamp'],
                'auroc': metrics_data.get('auroc'),
                'optimal_threshold': metrics_data.get('optimal_threshold'),
                'precision': metrics_data.get('precision'),
                'recall': metrics_data.get('recall'),
                'f1_score': metrics_data.get('f1_score'),
                'accuracy': accuracy,
                'full_dir_name': exp_info['full_name']
            })

        except Exception as e:
            print(f"   ⚠️ {exp_dir.name} metrics_report.json 로드 실패: {e}")

    if not experiment_results:
        print("❌ 분석할 수 있는 metrics_report.json 파일이 없습니다.")
        return

    # DataFrame 생성
    df = pd.DataFrame(experiment_results)

    # 정렬: 실험명 오름차순 -> 도메인 오름차순 -> 타임스탬프 오름차순
    df = df.sort_values(['experiment_name', 'domain', 'timestamp'], ascending=[True, True, True])

    # 결과 출력
    print(f"\n{'='*120}")
    print(f"🎯 Single-Domain 실험 결과")
    print(f"{'='*120}")
    print(f"총 실험 수: {len(df)}")

    print(f"\n📊 실험별 상세 성능:")

    # 출력용 DataFrame 생성
    display_df = df.copy()
    display_df['실험명'] = display_df['experiment_name']
    display_df['도메인'] = display_df['domain']
    display_df['타임스탬프'] = display_df['timestamp']
    display_df['AUROC'] = display_df['auroc'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else 'N/A')
    display_df['Threshold'] = display_df['optimal_threshold'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else 'N/A')
    display_df['Precision'] = display_df['precision'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else 'N/A')
    display_df['Recall'] = display_df['recall'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else 'N/A')
    display_df['F1 Score'] = display_df['f1_score'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else 'N/A')
    display_df['Accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else 'N/A')

    # 출력할 칼럼만 선택
    output_columns = ['실험명', '도메인', '타임스탬프', 'AUROC', 'Threshold', 'Precision', 'Recall', 'F1 Score', 'Accuracy']
    display_df = display_df[output_columns]

    # CSV 형태로 출력 (comma separated)
    print(display_df.to_csv(index=False, lineterminator='\n'))

    # CSV 저장
    output_path = results_path / "single_domain_analysis.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장됨: {output_path}")

    # ========================================================================
    # 두 번째 View: 실험별 도메인 반복 횟수
    # ========================================================================
    print(f"\n{'='*120}")
    print(f"📊 실험별 도메인 반복 횟수")
    print(f"{'='*120}")

    # 실험 번호 추출 (exp-71.A -> exp-71)
    def extract_experiment_number(exp_name):
        if '.' in exp_name and exp_name.startswith('exp-'):
            # exp-71.A -> exp-71
            return exp_name.split('.')[0]
        return exp_name

    df['experiment_number'] = df['experiment_name'].apply(extract_experiment_number)

    # 실험 번호별, 도메인별 반복 횟수 계산
    count_data = []

    for exp_num in sorted(df['experiment_number'].unique()):
        exp_data = df[df['experiment_number'] == exp_num]

        row_data = {'실험명': exp_num}

        # 각 도메인에 대해 반복 횟수 계산
        for domain in ['A', 'B', 'C', 'D']:
            domain_count = len(exp_data[exp_data['domain'] == domain])
            row_data[f'Domain_{domain}'] = domain_count

        count_data.append(row_data)

    # DataFrame 생성
    count_df = pd.DataFrame(count_data)

    # CSV 형태로 출력
    print(count_df.to_csv(index=False, lineterminator='\n'))

    # CSV 저장
    count_output_path = results_path / "experiment_domain_count.csv"
    count_df.to_csv(count_output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 실험별 도메인 반복 횟수 저장됨: {count_output_path}")

    # ========================================================================
    # 세 번째 View: 실험별 도메인 Accuracy 평균/표준편차 매트릭스
    # ========================================================================
    print(f"\n{'='*120}")
    print(f"📊 실험별 도메인 Accuracy 평균/표준편차")
    print(f"{'='*120}")

    # 실험 번호 추출 (exp-71.A -> exp-71)
    def extract_experiment_number(exp_name):
        if '.' in exp_name and exp_name.startswith('exp-'):
            # exp-71.A -> exp-71
            return exp_name.split('.')[0]
        return exp_name

    df['experiment_number'] = df['experiment_name'].apply(extract_experiment_number)

    # 실험 번호별, 도메인별 accuracy 평균과 표준편차 계산
    summary_data = []

    for exp_num in sorted(df['experiment_number'].unique()):
        exp_data = df[df['experiment_number'] == exp_num]

        row_data = {'실험명': exp_num}

        # 각 도메인에 대해 평균과 표준편차 계산
        for domain in ['A', 'B', 'C', 'D']:
            domain_data = exp_data[exp_data['domain'] == domain]['accuracy']

            if len(domain_data) > 0:
                mean_acc = domain_data.mean()
                # 데이터가 1개면 std = 0
                std_acc = domain_data.std() if len(domain_data) > 1 else 0.0

                row_data[f'Domain_{domain}_Mean'] = f"{mean_acc:.6f}"
                row_data[f'Domain_{domain}_Std'] = f"{std_acc:.6f}"
            else:
                row_data[f'Domain_{domain}_Mean'] = 'N/A'
                row_data[f'Domain_{domain}_Std'] = 'N/A'

        summary_data.append(row_data)

    # DataFrame 생성
    summary_df = pd.DataFrame(summary_data)

    # CSV 형태로 출력
    print(summary_df.to_csv(index=False, lineterminator='\n'))

    # CSV 저장
    summary_output_path = results_path / "experiment_domain_accuracy_summary.csv"
    summary_df.to_csv(summary_output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 실험별 도메인 Accuracy 요약 저장됨: {summary_output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Single-Domain 실험 결과 분석')
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='실험 결과 디렉토리 경로 (예: results_100k_train)'
    )

    args = parser.parse_args()

    # 분석 실행
    analyze_single_domain_experiments(args.results_dir)


if __name__ == "__main__":
    main()
