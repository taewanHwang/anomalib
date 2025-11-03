import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set font to Computer Modern (LaTeX 기본 폰트)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['cmr10', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.formatter.use_mathtext'] = True

# Severity levels
severity = np.array([10, 30, 50, 100, 200])

# Scaled CutPaste (opt) == cutpaste clf (conventional)
scp_means = np.array([0.54712925, 0.59166675, 0.9403705, 0.94578725, 0.9830095]) * 100
scp_stds  = np.array([0.0487625, 0.079146, 0.07507825, 0.076159, 0.0079345]) * 100

# Proposed (Draem+CP)
prop_means = np.array([0.99760425, 0.99874975, 0.9971665, 0.99482625, 0.99267375]) * 100
prop_stds  = np.array([0.001519, 0.0001965, 0.00157775, 0.0027015, 0.001981]) * 100

plt.figure(figsize=(10,6))

plt.errorbar(severity, scp_means, yerr=scp_stds, marker='o', linewidth=2, capsize=5, label="Scaled CutPaste")
plt.errorbar(severity, prop_means, yerr=prop_stds, marker='s', linewidth=2, capsize=5, label="Proposed")

plt.title("Robustness to fault intensity parameter ")
plt.xlabel("Severity change (%)")
plt.ylabel("Accuracy (%)")
plt.xticks(severity)
plt.ylim(40, 101)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()

plt.savefig("accruacy_over_severity_change.png", dpi=300, bbox_inches="tight")
plt.show()
