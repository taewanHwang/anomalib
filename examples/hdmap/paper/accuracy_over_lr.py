import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set font to Computer Modern (LaTeX 기본 폰트)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['cmr10', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.formatter.use_mathtext'] = True

# Learning rates in desired order
lrs = np.array([1e-4, 5e-4, 1e-3])
lr_labels = ["1e-4", "5e-4", "1e-3"]

# Domain mean and std for Scaled CutPaste
scaled_mean = np.array([99.47, 94.58, 81.52])
scaled_std  = np.array([0.11, 7.62, 7.84])

# Domain mean and std for Proposed
prop_mean = np.array([99.72, 99.86, 99.72])
prop_std  = np.array([0.31, 0.06, 0.18])

plt.figure(figsize=(10,6))

# Plot Scaled CutPaste
plt.errorbar(lrs, scaled_mean, yerr=scaled_std, marker='o', linewidth=2, capsize=5, label="Scaled CutPaste")

# Plot Proposed
plt.errorbar(lrs, prop_mean, yerr=prop_std, marker='o', linewidth=2, capsize=5, label="Proposed")

plt.title("Robustness to learning rate variation")
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy (%)")
plt.xscale("log")
plt.xticks(lrs, lr_labels)
plt.ylim(80, 100)  # Show variability clearly
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.savefig("accuracy_over_lr.png", dpi=300, bbox_inches="tight")

plt.show()
