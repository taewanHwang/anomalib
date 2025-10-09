"""Plot test results: Anomaly Score and Ground Truth.

This script reads a CSV file and generates plots for anomaly scores and ground truth.
The CSV file path is hardcoded below.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# ============================================================================
# CSV 파일 경로 (절대 경로로 하드코딩)
# ============================================================================
# CSV_PATH = "/mnt/ex-disk/taewan.hwang/study/anomalib/results_exp16/20251009_124553/exp-16.v3.C_20251009_124553/analysis/test_results.csv"
CSV_PATH = "/mnt/ex-disk/taewan.hwang/study/anomalib/results_exp16/20251009_142358/exp-16-9.C_20251009_142358/analysis/test_results.csv"
# ============================================================================

csv_path = Path(CSV_PATH)
if not csv_path.exists():
    print(f"Error: CSV file not found: {csv_path}")
    print(f"Please update the CSV_PATH variable in the script.")
    exit(1)

# CSV 파일의 디렉토리
csv_dir = csv_path.parent

# CSV 읽기
df = pd.read_csv(csv_path)
print(f"Reading CSV from: {csv_path}")

print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# metrics_report.json에서 optimal threshold 읽기
metrics_path = csv_dir / "metrics_report.json"
optimal_threshold = None
if metrics_path.exists():
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    optimal_threshold = metrics.get('optimal_threshold')
    print(f"\nLoaded optimal threshold: {optimal_threshold:.6f}")
else:
    print(f"\nWarning: metrics_report.json not found at {metrics_path}")

# 분류 오류 찾기 (optimal threshold가 있는 경우)
false_positives = []  # Normal인데 threshold 초과
false_negatives = []  # Anomaly인데 threshold 미만
if optimal_threshold is not None:
    # False Positives: ground_truth = 0 (Normal)인데 score > threshold
    fp_mask = (df['ground_truth'] == 0) & (df['anomaly_score'] > optimal_threshold)
    false_positives = df[fp_mask].index.tolist()

    # False Negatives: ground_truth = 1 (Anomaly)인데 score < threshold
    fn_mask = (df['ground_truth'] == 1) & (df['anomaly_score'] < optimal_threshold)
    false_negatives = df[fn_mask].index.tolist()

    print(f"\nMisclassifications:")
    print(f"  False Positives (Normal classified as Anomaly): {len(false_positives)}")
    print(f"  False Negatives (Anomaly classified as Normal): {len(false_negatives)}")
    print(f"  Total Misclassifications: {len(false_positives) + len(false_negatives)}")

# Figure 생성
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# 1. Anomaly Score line chart
ax1.plot(df.index, df['anomaly_score'], linewidth=0.8, color='blue', alpha=0.7, label='Anomaly Score')
ax1.set_xlabel('Sample Index', fontsize=12)
ax1.set_ylabel('Anomaly Score', fontsize=12)
ax1.set_title('Anomaly Score by Sample Index', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# optimal threshold 선 추가
if optimal_threshold is not None:
    ax1.axhline(y=optimal_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Optimal Threshold: {optimal_threshold:.4f}')

    # 오분류 데이터 마커 추가
    if false_positives:
        ax1.scatter(false_positives, df.loc[false_positives, 'anomaly_score'],
                   marker='x', s=100, color='orange', linewidths=2,
                   label=f'False Positives: {len(false_positives)}', zorder=5)

    if false_negatives:
        ax1.scatter(false_negatives, df.loc[false_negatives, 'anomaly_score'],
                   marker='x', s=100, color='purple', linewidths=2,
                   label=f'False Negatives: {len(false_negatives)}', zorder=5)

    ax1.legend(loc='upper right')

# 2. Ground Truth line chart
ax2.plot(df.index, df['ground_truth'], linewidth=0.8, color='green', alpha=0.7, marker='o', markersize=2)
ax2.set_xlabel('Sample Index', fontsize=12)
ax2.set_ylabel('Ground Truth', fontsize=12)
ax2.set_title('Ground Truth by Sample Index', fontsize=14, fontweight='bold')
ax2.set_ylim([-0.1, 1.1])
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['Normal (0)', 'Anomaly (1)'])
ax2.grid(True, alpha=0.3)

# Ground truth 영역 색칠
normal_mask = df['ground_truth'] == 0
anomaly_mask = df['ground_truth'] == 1
ax2.fill_between(df.index, 0, 1, where=normal_mask, alpha=0.2, color='lightgreen', label='Normal')
ax2.fill_between(df.index, 0, 1, where=anomaly_mask, alpha=0.2, color='lightcoral', label='Anomaly')
ax2.legend(loc='upper right')

plt.tight_layout()

# 저장 (CSV와 같은 디렉토리)
output_path = csv_dir / "test_results_plot.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Plot saved to: {output_path}")

# 통계 출력
print(f"\nStatistics:")
print(f"  Anomaly Score:")
print(f"    Min: {df['anomaly_score'].min():.6f}")
print(f"    Max: {df['anomaly_score'].max():.6f}")
print(f"    Mean: {df['anomaly_score'].mean():.6f}")
print(f"  Ground Truth:")
print(f"    Normal (0): {(df['ground_truth']==0).sum()}")
print(f"    Anomaly (1): {(df['ground_truth']==1).sum()}")
print(f"  Predicted Label:")
print(f"    Normal (0): {(df['predicted_label']==0).sum()}")
print(f"    Anomaly (1): {(df['predicted_label']==1).sum()}")
