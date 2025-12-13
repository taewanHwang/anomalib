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

# X-axis labels (fault shape levels)
shapes = ["Small", "Medium", "Large", "Full", "Extended"]

# Scaled CutPaste (conventional)
scp_means = np.array([96.0, 97.3, 96.9, 97.0, 94.0])
scp_stds  = np.array([1.7, 1.3, 1.6, 1.6, 2.8])

# Proposed (Draem+CP)
prop_means = np.array([97.1, 98.6, 98.6, 99.4, 99.3])
prop_stds  = np.array([3.4, 1.1, 1.2, 0.8, 0.9])

x = np.arange(len(shapes))

plt.figure(figsize=(10,6))

plt.errorbar(x, prop_means, yerr=prop_stds, marker=markers[1], linewidth=2,
             linestyle='-', color=colors[0], markersize=markersizes[1],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Proposed")

# Scaled CutPaste - error bar를 점선으로
scp_line = plt.errorbar(x, scp_means, yerr=scp_stds, marker=markers[0], linewidth=2,
                        linestyle=(0, (1, 1)), color=colors[1], markersize=markersizes[0],
                        capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Scaled CutPaste")
# error bar 선도 점선으로 설정
scp_line[-1][0].set_linestyle((0, (1, 1)))

plt.title("Robustness to fault shape variation", fontsize=16)
plt.xlabel("Fault shape", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xticks(x, shapes)
plt.ylim(90, 100)
plt.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5, color='gray')
plt.legend(fontsize=12, labelspacing=1.0)

plt.savefig("accuracy_over_fault_shape.png", dpi=300, bbox_inches="tight")
plt.show()
