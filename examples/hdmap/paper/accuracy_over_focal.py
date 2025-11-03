import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set font to Computer Modern (LaTeX 기본 폰트)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['cmr10', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.formatter.use_mathtext'] = True

# Focal alpha values
alphas = np.array([0.1, 0.5, 0.9, 1.0])

# Domain Average (mean ± std) for Proposed
avg_means = np.array([99.858, 99.339, 99.717, 88.153])
avg_stds  = np.array([0.057, 1.017, 0.158, 8.253])

# Domain C (mean ± std) for Proposed
c_means = np.array([99.600, 97.522, 99.222, 80.700])
c_stds  = np.array([0.227, 4.067, 0.239, 17.286])

plt.figure(figsize=(10,6))

plt.errorbar(alphas, avg_means, yerr=avg_stds, marker='o', linewidth=2, capsize=5, label="Domain Average")
plt.errorbar(alphas, c_means, yerr=c_stds, marker='s', linewidth=2, capsize=5, label="Domain C")

plt.title("Effect of Focal Loss Weight")
plt.xlabel("Focal alpha")
plt.ylabel("Accuracy (%)")
plt.xticks(alphas, [str(a) for a in alphas])
plt.ylim(75, 100.5)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()

# Save figure for download
plt.savefig("accuracy_over_focal.png", dpi=300, bbox_inches="tight")

plt.show()