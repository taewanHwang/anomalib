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

# Learning rates in desired order
lrs = np.array([1e-4, 5e-4, 1e-3])
# lr_labels = ["1e-4", "5e-4", "1e-3"]
lr_labels = [r"$10^{-4}$", r"$5 \times 10^{-4}$", r"$10^{-3}$"]

# Domain mean and std for Scaled CutPaste
scaled_mean = np.array([97.0, 97.0, 94.5])
scaled_std  = np.array([1.5, 1.6, 5.9])

# Domain mean and std for Proposed - Test
prop_test_mean = np.array([98.5, 99.4, 99.2])
prop_test_std  = np.array([2.8, 0.8, 0.7])

# Domain mean and std for Proposed - Validation
prop_val_mean = np.array([99.1, 99.3, 98.3])
prop_val_std  = np.array([0.2, 0.1, 0.3])

plt.figure(figsize=(10,6))

# Plot Proposed (test)
plt.errorbar(lrs, prop_test_mean, yerr=prop_test_std, marker=markers[1], linewidth=2,
             linestyle='-', color=colors[0], markersize=markersizes[1],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Proposed (test)")

# Plot Proposed (validation)
plt.errorbar(lrs, prop_val_mean, yerr=prop_val_std, marker=markers[1], linewidth=2,
             linestyle=':', color=colors[0], markersize=markersizes[1],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Proposed (validation)")

# Scaled CutPaste (test)
plt.errorbar(lrs, scaled_mean, yerr=scaled_std, marker=markers[0], linewidth=2,
             linestyle='-', color=colors[1], markersize=markersizes[0],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Scaled CutPaste (test)")

plt.title("Robustness to learning rate variation", fontsize=16)
plt.xlabel("Learning rate", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xscale("log")
plt.xticks(lrs, lr_labels)
plt.ylim(88, 100)  # Show variability clearly
plt.grid(axis='both', linestyle='--', alpha=0.3, linewidth=0.5, color='gray')
plt.legend(fontsize=12, labelspacing=1.0)
plt.savefig("accuracy_over_lr.png", dpi=300, bbox_inches="tight")

plt.show()
