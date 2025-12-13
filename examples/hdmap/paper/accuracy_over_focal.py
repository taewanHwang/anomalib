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

# Focal alpha values
alphas = np.array([0.1, 0.5, 0.9, 1.0])

# Domain Average (mean ± std) for Proposed
avg_means = np.array([99.1, 98.4, 97.7, 50])
avg_stds  = np.array([1.3, 0.8, 1.7, 0])

# Domain B (mean ± std) for Proposed
b_means = np.array([99.7, 99.7, 96.4, 50])
b_stds  = np.array([0.2, 0.2, 2.4, 0])


# Domain C (mean ± std) for Proposed
c_means = np.array([96.9, 98.2, 98.5, 50])
c_stds  = np.array([0.6, 0.6, 0.4, 0])

plt.figure(figsize=(10,6))

plt.errorbar(alphas, avg_means, yerr=avg_stds, marker=markers[0], linewidth=2,
             linestyle='-', color=colors[0], markersize=markersizes[0],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Domain Average")

# Domain B - error bar를 대시선으로
# b_line = plt.errorbar(alphas, b_means, yerr=b_stds, marker=markers[1], linewidth=2,
#                       linestyle=(0, (5, 5)), color=colors[1], markersize=markersizes[1],
#                       capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Domain B")
# error bar 선도 대시선으로 설정
# b_line[-1][0].set_linestyle((0, (5, 5)))
c_line = plt.errorbar(alphas, c_means, yerr=c_stds, marker=markers[2], linewidth=2,
             linestyle=':', color=colors[1], markersize=markersizes[2],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Domain C")
c_line[-1][0].set_linestyle(':')

plt.title("Effect of Focal Loss Weight", fontsize=16)
plt.xlabel("Focal alpha", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xticks(alphas, [str(a) for a in alphas])
plt.ylim(90, 100.5)
plt.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5, color='gray')
plt.legend(fontsize=12, labelspacing=1.0)

# Save figure for download
plt.savefig("accuracy_over_focal.png", dpi=300, bbox_inches="tight")

plt.show()