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
shapes = [r"Small" + "\n" + r"($w \leq 16, h \leq 8$)", 
          r"Medium" + "\n" + r"($w \leq 32, h \leq 8$)", 
          r"Large" + "\n" + r"($w \leq 64, h \leq 8$)", 
          r"Full" + "\n" + r"($w \leq 127, h \leq 8$)", 
          r"Extended" + "\n" + r"($w \leq 127, h \leq 12$)"]

# Scaled CutPaste (conventional)
scp_means = np.array([96.0, 97.3, 96.9, 97.0, 94.0])
scp_stds  = np.array([1.7, 1.3, 1.6, 1.6, 2.8])

# Proposed (Draem+CP) - Test
prop_test_means = np.array([97.1, 98.6, 98.6, 99.4, 99.3])
prop_test_stds  = np.array([3.4, 1.1, 1.2, 0.8, 0.9])

# Proposed (Draem+CP) - Validation
prop_val_means = np.array([98.8, 99.1, 99.1, 99.3, 98.9])
prop_val_stds  = np.array([0.2, 0.2, 0.3, 0.1, 0.2])

x = np.arange(len(shapes))

plt.figure(figsize=(10,6))

plt.errorbar(x, prop_test_means, yerr=prop_test_stds, marker=markers[1], linewidth=2,
             linestyle='-', color=colors[0], markersize=markersizes[1],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Proposed (test)")

plt.errorbar(x, prop_val_means, yerr=prop_val_stds, marker=markers[1], linewidth=2,
             linestyle=':', color=colors[0], markersize=markersizes[1],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Proposed (validation)")

# Scaled CutPaste (test)
plt.errorbar(x, scp_means, yerr=scp_stds, marker=markers[0], linewidth=2,
             linestyle='-', color=colors[1], markersize=markersizes[0],
             capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.8, label="Scaled CutPaste (test)")

plt.title("Robustness to patch geometry variation", fontsize=16)
plt.xlabel("Patch geometry", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xticks(x, shapes)
plt.ylim(90, 100)
plt.grid(axis='both', linestyle='--', alpha=0.3, linewidth=0.5, color='gray')
plt.legend(fontsize=12, labelspacing=1.0)

plt.savefig("accuracy_over_fault_shape.png", dpi=300, bbox_inches="tight")
plt.show()
