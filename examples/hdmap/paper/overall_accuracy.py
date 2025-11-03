import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set font to Computer Modern (LaTeX 기본 폰트)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['cmr10', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.formatter.use_mathtext'] = True

methods = ["DRAEM", "AD", "PatchCore", "Dinomaly", "Scaled \nCutPaste", "Scaled \nCutPaste \n(opt)", "Proposed"]
domain_avg = np.array([51.0, 70.6, 71.3, 93.0, 97.1, 99.5, 99.9])
domain_c    = np.array([50.0, 75.0, 60.9, 85.3, 93.3, 98.0, 99.7])

plt.figure(figsize=(10,6))

plt.plot(methods, domain_avg, marker='o', linewidth=2, label="Domain Average")
plt.plot(methods, domain_c, marker='o', linewidth=2, label="Domain C")

plt.title("Method Performance Comparison", fontsize=16)
plt.xlabel("Methods", fontsize=14)
plt.ylabel("Performance (%)", fontsize=14)
plt.ylim(45, 105)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# Save to paper directory
from pathlib import Path
output_path = Path(__file__).parent / "overall_accuracy.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"✅ Saved: {output_path}")

# plt.show()  # Commented out for non-interactive environments