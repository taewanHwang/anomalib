import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams

# Set font to Computer Modern (LaTeX 기본 폰트)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['cmr10', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.formatter.use_mathtext'] = True


# X-axis labels (fault shape levels)
shapes = ["smallest", "medium", "large", "largest"]

# Scaled CutPaste (conventional)
scp_means = np.array([97.722, 98.500, 98.407, 94.579])
scp_stds  = np.array([1.975, 0.521, 0.634, 7.616])

# Proposed (Draem+CP)
prop_means = np.array([99.092, 99.431, 99.431, 99.717])
prop_stds  = np.array([0.603, 0.202, 0.713, 0.158])

x = np.arange(len(shapes))

plt.figure(figsize=(10,6))

plt.errorbar(x, scp_means, yerr=scp_stds, marker='o', linewidth=2, capsize=5, label="Scaled CutPaste")
plt.errorbar(x, prop_means, yerr=prop_stds, marker='s', linewidth=2, capsize=5, label="Proposed")

plt.title("Robustness to fault shape variation")
plt.xlabel("Fault shape level")
plt.ylabel("Accuracy (%)")
plt.xticks(x, shapes)
plt.ylim(90, 100)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()

plt.savefig("accuracy_over_fault_shape.png", dpi=300, bbox_inches="tight")
plt.show()
