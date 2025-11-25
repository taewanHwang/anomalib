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
            val_metrics = metrics_data.get('validation_metrics')
            val_accuracy = (
                calculate_accuracy(val_metrics.get('confusion_matrix'))
                if val_metrics and val_metrics.get('confusion_matrix')
                else None
            )

            # Enhanced metrics 추출
            enhanced_metrics = metrics_data.get('enhanced_metrics')

            # 결과 저장
            result_dict = {
                'experiment_name': exp_info['experiment_name'],
                'domain': exp_info['domain'],
                'timestamp': exp_info['timestamp'],
                'auroc': metrics_data.get('auroc'),
                'optimal_threshold': metrics_data.get('optimal_threshold'),
                'precision': metrics_data.get('precision'),
                'recall': metrics_data.get('recall'),
                'f1_score': metrics_data.get('f1_score'),
                'accuracy': accuracy,
                'val_auroc': val_metrics.get('auroc') if val_metrics else None,
                'val_precision': val_metrics.get('precision') if val_metrics else None,
                'val_recall': val_metrics.get('recall') if val_metrics else None,
                'val_f1_score': val_metrics.get('f1_score') if val_metrics else None,
                'val_accuracy': val_accuracy,
                'full_dir_name': exp_info['full_name']
            }

            # Enhanced metrics 추가 (있는 경우)
            if enhanced_metrics:
                result_dict.update({
                    'enhanced_auroc': enhanced_metrics.get('auroc'),
                    # 99.5 percentile metrics
                    'percentile_99_5_threshold': enhanced_metrics.get('percentile_99_5_threshold'),
                    'tpr_at_percentile_99_5': enhanced_metrics.get('tpr_at_percentile_99_5'),
                    'fpr_at_percentile_99_5': enhanced_metrics.get('fpr_at_percentile_99_5'),
                    'precision_at_percentile_99_5': enhanced_metrics.get('precision_at_percentile_99_5'),
                    # 99.9 percentile metrics
                    'percentile_99_9_threshold': enhanced_metrics.get('percentile_99_9_threshold'),
                    'tpr_at_percentile_99_9': enhanced_metrics.get('tpr_at_percentile_99_9'),
                    'fpr_at_percentile_99_9': enhanced_metrics.get('fpr_at_percentile_99_9'),
                    'precision_at_percentile_99_9': enhanced_metrics.get('precision_at_percentile_99_9'),
                    # Fixed threshold metrics
                    'tpr_at_fixed': enhanced_metrics.get('tpr_at_fixed'),
                    'fpr_at_fixed': enhanced_metrics.get('fpr_at_fixed'),
                    'precision_at_fixed': enhanced_metrics.get('precision_at_fixed'),
                    'val_normal_score_mean': enhanced_metrics.get('val_normal_score_mean'),
                    'val_normal_score_std': enhanced_metrics.get('val_normal_score_std'),
                })

            experiment_results.append(result_dict)

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
    display_df['AUROC'] = display_df['auroc'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
    display_df['Threshold'] = display_df['optimal_threshold'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
    display_df['Precision'] = display_df['precision'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
    display_df['Recall'] = display_df['recall'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
    display_df['F1 Score'] = display_df['f1_score'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
    display_df['Test Accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
    display_df['Val AUROC'] = display_df['val_auroc'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
    display_df['Val Precision'] = display_df['val_precision'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
    display_df['Val Recall'] = display_df['val_recall'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
    display_df['Val F1 Score'] = display_df['val_f1_score'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
    display_df['Val Accuracy'] = display_df['val_accuracy'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')

    # Enhanced metrics 컬럼 추가 (있는 경우)
    if 'tpr_at_percentile_99_5' in df.columns:
        # 99.5 percentile
        display_df['TPR@99.5%'] = df['tpr_at_percentile_99_5'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
        display_df['Prec@99.5%'] = df['precision_at_percentile_99_5'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
        display_df['Thresh@99.5%'] = df['percentile_99_5_threshold'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
        # 99.9 percentile
        display_df['TPR@99.9%'] = df['tpr_at_percentile_99_9'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
        display_df['Prec@99.9%'] = df['precision_at_percentile_99_9'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
        display_df['Thresh@99.9%'] = df['percentile_99_9_threshold'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
        # Fixed 0.5
        display_df['TPR@0.5'] = df['tpr_at_fixed'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
        display_df['Prec@0.5'] = df['precision_at_fixed'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')

    # 출력할 칼럼만 선택
    output_columns = [
        '실험명', '도메인', '타임스탬프',
        'AUROC', 'Threshold', 'Precision', 'Recall', 'F1 Score', 'Test Accuracy',
        'Val AUROC', 'Val Precision', 'Val Recall', 'Val F1 Score', 'Val Accuracy'
    ]

    # Enhanced metrics 컬럼 추가 (있는 경우)
    if 'TPR@99.5%' in display_df.columns:
        output_columns.extend([
            'TPR@99.5%', 'Prec@99.5%', 'Thresh@99.5%',
            'TPR@99.9%', 'Prec@99.9%', 'Thresh@99.9%',
            'TPR@0.5', 'Prec@0.5'
        ])

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

    summary_rows = []
    val_summary_rows = []
    for exp_num in sorted(df['experiment_number'].unique()):
        exp_data = df[df['experiment_number'] == exp_num]
        row = {'실험명': exp_num}
        val_row = {'실험명': exp_num}

        for domain in ['A', 'B', 'C', 'D']:
            test_data = exp_data[exp_data['domain'] == domain]['accuracy']
            if len(test_data) > 0:
                mean_test = test_data.mean()
                std_test = test_data.std() if len(test_data) > 1 else 0.0
                row[f'Domain_{domain}_Mean'] = f"{mean_test:.4f}"
                row[f'Domain_{domain}_Std'] = f"{std_test:.4f}"
            else:
                row[f'Domain_{domain}_Mean'] = 'N/A'
                row[f'Domain_{domain}_Std'] = 'N/A'

            val_data = exp_data[exp_data['domain'] == domain]['val_accuracy']
            if len(val_data) > 0:
                mean_val = val_data.mean()
                std_val = val_data.std() if len(val_data) > 1 else 0.0
                val_row[f'Val_Domain_{domain}_Mean'] = f"{mean_val:.4f}"
                val_row[f'Val_Domain_{domain}_Std'] = f"{std_val:.4f}"
            else:
                val_row[f'Val_Domain_{domain}_Mean'] = 'N/A'
                val_row[f'Val_Domain_{domain}_Std'] = 'N/A'

        summary_rows.append(row)
        val_summary_rows.append(val_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df['Overall_Mean'] = df.groupby('experiment_number')['accuracy'].mean().apply(lambda x: f"{x:.4f}").values
    summary_df['Overall_Std'] = df.groupby('experiment_number')['accuracy'].std().fillna(0).apply(lambda x: f"{x:.4f}").values
    print(summary_df.to_csv(index=False, lineterminator='\n'))
    summary_output_path = results_path / "experiment_domain_accuracy_summary.csv"
    summary_df.to_csv(summary_output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 실험별 도메인 Test Accuracy 요약 저장됨: {summary_output_path}")

    val_summary_df = pd.DataFrame(val_summary_rows)
    val_summary_df['Val_Overall_Mean'] = df.groupby('experiment_number')['val_accuracy'].mean().apply(lambda x: f"{x:.4f}").values
    val_summary_df['Val_Overall_Std'] = df.groupby('experiment_number')['val_accuracy'].std().fillna(0).apply(lambda x: f"{x:.4f}").values
    print(val_summary_df.to_csv(index=False, lineterminator='\n'))
    val_summary_output_path = results_path / "experiment_domain_val_accuracy_summary.csv"
    val_summary_df.to_csv(val_summary_output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 실험별 도메인 Validation Accuracy 요약 저장됨: {val_summary_output_path}")

    # ========================================================================
    # 네 번째 View: Enhanced Metrics 요약 (TPR, Precision at different thresholds)
    # ========================================================================
    if 'tpr_at_percentile_99_5' in df.columns:
        print(f"\n{'='*120}")
        print(f"📊 실험별 Enhanced Metrics 요약 (TPR & Precision @ Different Thresholds)")
        print(f"{'='*120}")

        enhanced_summary_rows = []
        for exp_num in sorted(df['experiment_number'].unique()):
            exp_data = df[df['experiment_number'] == exp_num]
            row = {'실험명': exp_num}

            for domain in ['A', 'B', 'C', 'D']:
                domain_data = exp_data[exp_data['domain'] == domain]

                # TPR @ 99.5 percentile
                tpr_995_data = domain_data['tpr_at_percentile_99_5']
                if len(tpr_995_data) > 0:
                    mean_tpr = tpr_995_data.mean()
                    std_tpr = tpr_995_data.std() if len(tpr_995_data) > 1 else 0.0
                    row[f'Domain_{domain}_TPR@99.5%_Mean'] = f"{mean_tpr:.4f}"
                    row[f'Domain_{domain}_TPR@99.5%_Std'] = f"{std_tpr:.4f}"
                else:
                    row[f'Domain_{domain}_TPR@99.5%_Mean'] = 'N/A'
                    row[f'Domain_{domain}_TPR@99.5%_Std'] = 'N/A'

                # Precision @ 99.5 percentile
                prec_995_data = domain_data['precision_at_percentile_99_5']
                if len(prec_995_data) > 0:
                    mean_prec = prec_995_data.mean()
                    std_prec = prec_995_data.std() if len(prec_995_data) > 1 else 0.0
                    row[f'Domain_{domain}_Prec@99.5%_Mean'] = f"{mean_prec:.4f}"
                    row[f'Domain_{domain}_Prec@99.5%_Std'] = f"{std_prec:.4f}"
                else:
                    row[f'Domain_{domain}_Prec@99.5%_Mean'] = 'N/A'
                    row[f'Domain_{domain}_Prec@99.5%_Std'] = 'N/A'

                # TPR @ 99.9 percentile
                tpr_999_data = domain_data['tpr_at_percentile_99_9']
                if len(tpr_999_data) > 0:
                    mean_tpr = tpr_999_data.mean()
                    std_tpr = tpr_999_data.std() if len(tpr_999_data) > 1 else 0.0
                    row[f'Domain_{domain}_TPR@99.9%_Mean'] = f"{mean_tpr:.4f}"
                    row[f'Domain_{domain}_TPR@99.9%_Std'] = f"{std_tpr:.4f}"
                else:
                    row[f'Domain_{domain}_TPR@99.9%_Mean'] = 'N/A'
                    row[f'Domain_{domain}_TPR@99.9%_Std'] = 'N/A'

                # Precision @ 99.9 percentile
                prec_999_data = domain_data['precision_at_percentile_99_9']
                if len(prec_999_data) > 0:
                    mean_prec = prec_999_data.mean()
                    std_prec = prec_999_data.std() if len(prec_999_data) > 1 else 0.0
                    row[f'Domain_{domain}_Prec@99.9%_Mean'] = f"{mean_prec:.4f}"
                    row[f'Domain_{domain}_Prec@99.9%_Std'] = f"{std_prec:.4f}"
                else:
                    row[f'Domain_{domain}_Prec@99.9%_Mean'] = 'N/A'
                    row[f'Domain_{domain}_Prec@99.9%_Std'] = 'N/A'

                # TPR @ Fixed 0.5
                tpr_fixed_data = domain_data['tpr_at_fixed']
                if len(tpr_fixed_data) > 0:
                    mean_tpr_fixed = tpr_fixed_data.mean()
                    std_tpr_fixed = tpr_fixed_data.std() if len(tpr_fixed_data) > 1 else 0.0
                    row[f'Domain_{domain}_TPR@0.5_Mean'] = f"{mean_tpr_fixed:.4f}"
                    row[f'Domain_{domain}_TPR@0.5_Std'] = f"{std_tpr_fixed:.4f}"
                else:
                    row[f'Domain_{domain}_TPR@0.5_Mean'] = 'N/A'
                    row[f'Domain_{domain}_TPR@0.5_Std'] = 'N/A'

            enhanced_summary_rows.append(row)

        enhanced_summary_df = pd.DataFrame(enhanced_summary_rows)

        # Overall statistics
        enhanced_summary_df['TPR@99.5%_Overall_Mean'] = df.groupby('experiment_number')['tpr_at_percentile_99_5'].mean().apply(lambda x: f"{x:.4f}").values
        enhanced_summary_df['TPR@99.5%_Overall_Std'] = df.groupby('experiment_number')['tpr_at_percentile_99_5'].std().fillna(0).apply(lambda x: f"{x:.4f}").values
        enhanced_summary_df['Prec@99.5%_Overall_Mean'] = df.groupby('experiment_number')['precision_at_percentile_99_5'].mean().apply(lambda x: f"{x:.4f}").values
        enhanced_summary_df['Prec@99.5%_Overall_Std'] = df.groupby('experiment_number')['precision_at_percentile_99_5'].std().fillna(0).apply(lambda x: f"{x:.4f}").values
        enhanced_summary_df['TPR@99.9%_Overall_Mean'] = df.groupby('experiment_number')['tpr_at_percentile_99_9'].mean().apply(lambda x: f"{x:.4f}").values
        enhanced_summary_df['TPR@99.9%_Overall_Std'] = df.groupby('experiment_number')['tpr_at_percentile_99_9'].std().fillna(0).apply(lambda x: f"{x:.4f}").values
        enhanced_summary_df['Prec@99.9%_Overall_Mean'] = df.groupby('experiment_number')['precision_at_percentile_99_9'].mean().apply(lambda x: f"{x:.4f}").values
        enhanced_summary_df['Prec@99.9%_Overall_Std'] = df.groupby('experiment_number')['precision_at_percentile_99_9'].std().fillna(0).apply(lambda x: f"{x:.4f}").values
        enhanced_summary_df['TPR@0.5_Overall_Mean'] = df.groupby('experiment_number')['tpr_at_fixed'].mean().apply(lambda x: f"{x:.4f}").values
        enhanced_summary_df['TPR@0.5_Overall_Std'] = df.groupby('experiment_number')['tpr_at_fixed'].std().fillna(0).apply(lambda x: f"{x:.4f}").values

        print(enhanced_summary_df.to_csv(index=False, lineterminator='\n'))
        enhanced_summary_output_path = results_path / "experiment_enhanced_metrics_summary.csv"
        enhanced_summary_df.to_csv(enhanced_summary_output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 실험별 Enhanced Metrics 요약 저장됨: {enhanced_summary_output_path}")

    # ========================================================================
    # 다섯 번째 View: Threshold별 Test Accuracy 비교 (논문용 - 도메인별)
    # ========================================================================
    if 'tpr_at_percentile_99_5' in df.columns and 'fpr_at_percentile_99_5' in df.columns:
        print(f"\n{'='*120}")
        print(f"📊 논문용: Threshold별 Test Accuracy 비교 (Balanced Test Set)")
        print(f"{'='*120}")

        # Accuracy 계산 함수
        def calculate_accuracy_from_tpr_fpr(tpr, fpr):
            """
            Balanced test set에서 TPR과 FPR로부터 Accuracy 계산
            Accuracy = (TP + TN) / Total = (TPR + (1-FPR)) / 2  (when P=N, balanced)
            """
            return (tpr + (1 - fpr)) / 2

        # ========== Table 1: Accuracy @ 99.5%-ile Threshold ==========
        print(f"\n{'='*120}")
        print(f"📋 Table 1: Test Accuracy @ 99.5%-ile Threshold (from Validation)")
        print(f"{'='*120}")

        acc_995_rows = []
        for exp_num in sorted(df['experiment_number'].unique()):
            exp_data = df[df['experiment_number'] == exp_num]
            row = {'실험명': exp_num}

            for domain in ['A', 'B', 'C', 'D']:
                domain_data = exp_data[exp_data['domain'] == domain]
                if len(domain_data) > 0 and 'fpr_at_percentile_99_5' in domain_data.columns:
                    tpr = domain_data['tpr_at_percentile_99_5']
                    fpr = domain_data['fpr_at_percentile_99_5']
                    acc = (tpr + (1 - fpr)) / 2
                    mean_acc = acc.mean()
                    std_acc = acc.std() if len(acc) > 1 else 0.0
                    row[f'Domain_{domain}_Mean'] = f"{mean_acc:.4f}"
                    row[f'Domain_{domain}_Std'] = f"{std_acc:.4f}"
                else:
                    row[f'Domain_{domain}_Mean'] = 'N/A'
                    row[f'Domain_{domain}_Std'] = 'N/A'

            # Overall
            tpr_all = exp_data['tpr_at_percentile_99_5']
            fpr_all = exp_data['fpr_at_percentile_99_5']
            acc_all = (tpr_all + (1 - fpr_all)) / 2
            row['Overall_Mean'] = f"{acc_all.mean():.4f}"
            row['Overall_Std'] = f"{acc_all.std():.4f}" if len(acc_all) > 1 else "0.0000"

            acc_995_rows.append(row)

        acc_995_df = pd.DataFrame(acc_995_rows)
        print(acc_995_df.to_csv(index=False, lineterminator='\n'))
        acc_995_output_path = results_path / "accuracy_at_99_5_percentile.csv"
        acc_995_df.to_csv(acc_995_output_path, index=False, encoding='utf-8-sig')
        print(f"💾 저장됨: {acc_995_output_path}")

        # ========== Table 2: Accuracy @ 99.9%-ile Threshold ==========
        print(f"\n{'='*120}")
        print(f"📋 Table 2: Test Accuracy @ 99.9%-ile Threshold (from Validation)")
        print(f"{'='*120}")

        acc_999_rows = []
        for exp_num in sorted(df['experiment_number'].unique()):
            exp_data = df[df['experiment_number'] == exp_num]
            row = {'실험명': exp_num}

            for domain in ['A', 'B', 'C', 'D']:
                domain_data = exp_data[exp_data['domain'] == domain]
                if len(domain_data) > 0 and 'fpr_at_percentile_99_9' in domain_data.columns:
                    tpr = domain_data['tpr_at_percentile_99_9']
                    fpr = domain_data['fpr_at_percentile_99_9']
                    acc = (tpr + (1 - fpr)) / 2
                    mean_acc = acc.mean()
                    std_acc = acc.std() if len(acc) > 1 else 0.0
                    row[f'Domain_{domain}_Mean'] = f"{mean_acc:.4f}"
                    row[f'Domain_{domain}_Std'] = f"{std_acc:.4f}"
                else:
                    row[f'Domain_{domain}_Mean'] = 'N/A'
                    row[f'Domain_{domain}_Std'] = 'N/A'

            # Overall
            tpr_all = exp_data['tpr_at_percentile_99_9']
            fpr_all = exp_data['fpr_at_percentile_99_9']
            acc_all = (tpr_all + (1 - fpr_all)) / 2
            row['Overall_Mean'] = f"{acc_all.mean():.4f}"
            row['Overall_Std'] = f"{acc_all.std():.4f}" if len(acc_all) > 1 else "0.0000"

            acc_999_rows.append(row)

        acc_999_df = pd.DataFrame(acc_999_rows)
        print(acc_999_df.to_csv(index=False, lineterminator='\n'))
        acc_999_output_path = results_path / "accuracy_at_99_9_percentile.csv"
        acc_999_df.to_csv(acc_999_output_path, index=False, encoding='utf-8-sig')
        print(f"💾 저장됨: {acc_999_output_path}")

        # ========== Table 3: Accuracy @ Fixed 0.5 Threshold ==========
        print(f"\n{'='*120}")
        print(f"📋 Table 3: Test Accuracy @ Fixed 0.5 Threshold (No Validation)")
        print(f"{'='*120}")

        acc_05_rows = []
        for exp_num in sorted(df['experiment_number'].unique()):
            exp_data = df[df['experiment_number'] == exp_num]
            row = {'실험명': exp_num}

            for domain in ['A', 'B', 'C', 'D']:
                domain_data = exp_data[exp_data['domain'] == domain]
                if len(domain_data) > 0:
                    acc = domain_data['accuracy']
                    mean_acc = acc.mean()
                    std_acc = acc.std() if len(acc) > 1 else 0.0
                    row[f'Domain_{domain}_Mean'] = f"{mean_acc:.4f}"
                    row[f'Domain_{domain}_Std'] = f"{std_acc:.4f}"
                else:
                    row[f'Domain_{domain}_Mean'] = 'N/A'
                    row[f'Domain_{domain}_Std'] = 'N/A'

            # Overall
            acc_all = exp_data['accuracy']
            row['Overall_Mean'] = f"{acc_all.mean():.4f}"
            row['Overall_Std'] = f"{acc_all.std():.4f}" if len(acc_all) > 1 else "0.0000"

            acc_05_rows.append(row)

        acc_05_df = pd.DataFrame(acc_05_rows)
        print(acc_05_df.to_csv(index=False, lineterminator='\n'))
        acc_05_output_path = results_path / "accuracy_at_fixed_0_5.csv"
        acc_05_df.to_csv(acc_05_output_path, index=False, encoding='utf-8-sig')
        print(f"💾 저장됨: {acc_05_output_path}")

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
