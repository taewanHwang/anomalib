import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set font to Computer Modern (LaTeX 기본 폰트)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['cmr10', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.formatter.use_mathtext'] = True

# IEEE/Nature journal style
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']
markersizes = [6, 6, 7, 6]

methods = ["DRAEM","CutPaste","2D CNN","UniNet","Reverse\nDistillation","PaDiM","Dinomaly","PatchCore","CFlow-AD","Scaled\nCutPaste", "Proposed"]
domain_avg = np.array([51.0, 65.9, 70.6, 72.4 ,78.0, 81.7, 90.9, 91.6, 92.0, 97.1, 99.6])
domain_avg_std = np.array([2.0, 18.0, 0.8, 9.8, 8.3, 9.3, 5.7, 7.7, 6.2, 0.5, 0.4])

domain_c    = np.array([50, 72.1, 75, 73.3, 71.2, 72.9, 83.9, 79.9, 84.4, 93.3, 99.1])
domain_c_std = np.array([0, 15.3, 0, 1.3, 4.2, 1.6, 0.8, 1.2, 1.4, 1.0, 0.6])

x_positions = np.arange(len(methods))

plt.figure(figsize=(10,6))

plt.errorbar(x_positions, domain_avg, yerr=domain_avg_std, marker=markers[0], linewidth=2,
             linestyle='-', color=colors[0], markersize=markersizes[0],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Domain average")

# Domain C - error bar를 대시선으로
c_line = plt.errorbar(x_positions, domain_c, yerr=domain_c_std, marker=markers[1], linewidth=2,
                      linestyle='--', color=colors[1], markersize=markersizes[1],
                      capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Domain C")
# error bar 선도 대시선으로 설정
c_line[-1][0].set_linestyle('--')

plt.title("Overall fault detection accuracy", fontsize=16)
plt.xlabel("Method", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xticks(x_positions, methods)
plt.ylim(45, 105)
plt.grid(axis='both', linestyle='--', alpha=0.3, linewidth=0.5, color='gray')
plt.legend(fontsize=12, labelspacing=1.0)

# Save to paper directory
from pathlib import Path
output_path = Path(__file__).parent / "overall_accuracy.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"✅ Saved: {output_path}")

# plt.show()  # Commented out for non-interactive environments