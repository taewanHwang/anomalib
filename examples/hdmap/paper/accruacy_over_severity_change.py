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

# Severity levels
# severity = np.array([10, 30, 50, 100, 200, 300, 400, 500])
severity = np.array([10, 30, 50, 100, 200])

# Scaled CutPaste (opt) == cutpaste clf (conventional)
# scp_means = np.array([0.5, 0.581, 0.946, 0.945, 0.939, 0.970, 0.959, 0.921]) * 100
# scp_stds  = np.array([0.0, 0.105, 0.068, 0.084, 0.063, 0.016, 0.020, 0.074]) * 100
scp_means = np.array([0.5, 0.581, 0.946, 0.945, 0.939]) * 100
scp_stds  = np.array([0.0, 0.105, 0.068, 0.084, 0.063]) * 100


# Proposed (Draem+CP)
# prop_means = np.array([0.983, 0.995, 0.996, 0.989, 0.982, 0.971, 0.970, 0.964]) * 100
# prop_stds  = np.array([0.022, 0.005, 0.004, 0.005, 0.006, 0.014, 0.007, 0.005]) * 100
prop_means = np.array([0.983, 0.995, 0.996, 0.989, 0.982]) * 100
prop_stds  = np.array([0.022, 0.005, 0.004, 0.005, 0.006]) * 100


plt.figure(figsize=(10,6))

plt.errorbar(severity, prop_means, yerr=prop_stds, marker=markers[1], linewidth=2,
             linestyle='-', color=colors[0], markersize=markersizes[1],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Proposed")

# Scaled CutPaste - error bar를 점선으로
scp_line = plt.errorbar(severity, scp_means, yerr=scp_stds, marker=markers[0], linewidth=2,
                        linestyle=(0, (1, 1)), color=colors[1], markersize=markersizes[0],
                        capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Scaled CutPaste")
# error bar 선도 점선으로 설정
scp_line[-1][0].set_linestyle((0, (1, 1)))

plt.title("Robustness to fault intensity parameter", fontsize=16)
plt.xlabel("Severity change (%)", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xticks(severity)
plt.ylim(40, 101)
plt.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5, color='gray')
plt.legend(fontsize=12, labelspacing=1.0)

plt.savefig("accruacy_over_severity_change.png", dpi=300, bbox_inches="tight")
plt.show()
