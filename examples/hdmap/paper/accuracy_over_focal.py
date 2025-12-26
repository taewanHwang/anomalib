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

# Domain Average (mean ± std) for Proposed - Test
avg_test_means = np.array([99.1, 99.4, 97.7, 50])
avg_test_stds  = np.array([1.3, 0.8, 1.7, 0])

# Domain Average (mean ± std) for Proposed - Validation
avg_val_means = np.array([99.0, 99.3, 98.7, 50])
avg_val_stds  = np.array([0.2, 0.1, 0.3, 0])


# Domain C (mean ± std) for Proposed - Test
c_test_means = np.array([96.9, 98.2, 98.5, 50])
c_test_stds  = np.array([0.6, 0.6, 0.4, 0])

# Domain C (mean ± std) for Proposed - Validation
c_val_means = np.array([99.1, 99.4, 98.8, 50])
c_val_stds  = np.array([0.11, 0.05, 0.19, 0])

plt.figure(figsize=(10,6))

plt.errorbar(alphas, avg_test_means, yerr=avg_test_stds, marker=markers[0], linewidth=2,
             linestyle='-', color=colors[0], markersize=markersizes[0],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Domain Average (test)")

plt.errorbar(alphas, avg_val_means, yerr=avg_val_stds, marker=markers[0], linewidth=2,
             linestyle=':', color=colors[0], markersize=markersizes[0],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Domain Average (validation)")

plt.errorbar(alphas, c_test_means, yerr=c_test_stds, marker=markers[2], linewidth=2,
             linestyle='-', color=colors[1], markersize=markersizes[2],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Domain C (test)")

plt.errorbar(alphas, c_val_means, yerr=c_val_stds, marker=markers[2], linewidth=2,
             linestyle=':', color=colors[1], markersize=markersizes[2],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Domain C (validation)")

plt.title(r"Effect of focal loss weight ($\alpha$)", fontsize=16)
plt.xlabel(r"Focal loss weight $\alpha$", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xticks(alphas, [str(a) for a in alphas])
plt.ylim(90, 100.5)
plt.grid(axis='both', linestyle='--', alpha=0.3, linewidth=0.5, color='gray')
plt.legend(fontsize=12, labelspacing=1.0)

# Save figure for download
plt.savefig("accuracy_over_focal.png", dpi=300, bbox_inches="tight")

plt.show()