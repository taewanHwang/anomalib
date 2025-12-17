"""
RSE (Radial Spectral Entropy) Computation Process Visualization.

This script visualizes each step of RSE calculation:
1. Original Image
2. Grayscale conversion (if RGB)
3. 2D FFT Power Spectrum
4. Radial Distance Map with Bins
5. Radial Energy Distribution
6. Final RSE Value

Shows how energy is aggregated by radial distance (discrete binning).
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# Paths
HDMAP_PNG_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png")
OUTPUT_DIR = Path(__file__).parent / "results" / "rse_process"

# Dinomaly preprocessing
DINOMALY_TRANSFORM = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_rse_with_intermediates(
    image: torch.Tensor,
    num_radial_bins: int = 64,
    apply_hann_window: bool = True,
    remove_dc: bool = True,
    eps: float = 1e-12,
) -> dict:
    """
    Compute RSE and return all intermediate results for visualization.

    Returns dict with:
        - gray: grayscale image
        - gray_windowed: grayscale with Hann window applied
        - power_spectrum: 2D FFT power spectrum (shifted)
        - radial_map: distance from center for each pixel
        - bin_edges: radial bin edges
        - bin_idx_map: bin index for each pixel
        - radial_energy: energy per radial bin
        - radial_distribution: normalized probability distribution
        - rse: final RSE value
    """
    C, H, W = image.shape
    device = image.device

    results = {}

    # Convert to grayscale
    if C == 3:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        image_denorm = image * std + mean

        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(3, 1, 1)
        gray = (image_denorm * weights).sum(dim=0)  # (H, W)
    else:
        gray = image.squeeze(0)

    results['gray'] = gray.cpu().numpy()

    # Remove DC (subtract mean)
    if remove_dc:
        gray = gray - gray.mean()

    # Apply Hann window
    gray_windowed = gray.clone()
    if apply_hann_window:
        wy = torch.hann_window(H, device=device).view(H, 1)
        wx = torch.hann_window(W, device=device).view(1, W)
        window = wy * wx
        gray_windowed = gray * window

    results['gray_windowed'] = gray_windowed.cpu().numpy()

    # 2D FFT
    F_complex = torch.fft.fft2(gray_windowed)
    P = torch.abs(F_complex) ** 2
    P_shifted = torch.fft.fftshift(P)

    results['power_spectrum'] = P_shifted.cpu().numpy()

    # Build radial distance map
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    yy = torch.arange(H, device=device, dtype=torch.float32)
    xx = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    results['radial_map'] = rr.cpu().numpy()
    results['center'] = (cy, cx)

    r_max = rr.max().item()

    # Radial binning (DISCRETE bins)
    bin_edges = torch.linspace(0.0, r_max + eps, num_radial_bins + 1, device=device)
    bin_idx = torch.bucketize(rr.flatten(), bin_edges) - 1
    bin_idx = bin_idx.clamp(0, num_radial_bins - 1).view(H, W)

    results['bin_edges'] = bin_edges.cpu().numpy()
    results['bin_idx_map'] = bin_idx.cpu().numpy()
    results['num_bins'] = num_radial_bins

    # Compute radial energy per bin
    P_flat = P_shifted.flatten()
    bin_idx_flat = bin_idx.flatten()

    radial_energy = torch.zeros(num_radial_bins, device=device)
    radial_energy.scatter_add_(0, bin_idx_flat, P_flat)

    results['radial_energy'] = radial_energy.cpu().numpy()

    # Normalize to probability distribution
    total = radial_energy.sum()
    if total > 0:
        p = radial_energy / total
    else:
        p = torch.zeros_like(radial_energy)

    results['radial_distribution'] = p.cpu().numpy()

    # Shannon entropy (normalized)
    H_r = -torch.sum(p * torch.log(p + eps))
    H_norm = H_r / np.log(num_radial_bins)
    rse = H_norm.clamp(0.0, 1.0).item()

    results['rse'] = rse
    results['raw_entropy'] = H_r.item()
    results['max_entropy'] = np.log(num_radial_bins)

    return results


def visualize_rse_process(image_path: Path, output_path: Path, sample_label: str = ""):
    """Create comprehensive visualization of RSE computation."""

    # Load and preprocess image
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = DINOMALY_TRANSFORM(img_pil)

    # Compute RSE with intermediates
    results = compute_rse_with_intermediates(img_tensor, num_radial_bins=64)

    # Create figure with 6 subplots (2 rows x 3 cols)
    fig = plt.figure(figsize=(18, 12))

    # 1. Original Image
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(img_pil)
    ax1.set_title(f"1. Original Image\n{sample_label}", fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Grayscale (with Hann window)
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(results['gray_windowed'], cmap='gray')
    ax2.set_title("2. Grayscale + Hann Window\n(DC removed)", fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. 2D FFT Power Spectrum (log scale)
    ax3 = fig.add_subplot(2, 3, 3)
    power = results['power_spectrum']
    # Add small value for log scale
    power_log = np.log10(power + 1e-10)
    im3 = ax3.imshow(power_log, cmap='hot')
    ax3.set_title("3. 2D FFT Power Spectrum\n(log scale, shifted)", fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='log10(Power)')

    # Add center marker
    cy, cx = results['center']
    ax3.plot(cx, cy, 'c+', markersize=10, markeredgewidth=2)

    # 4. Radial Binning Visualization
    ax4 = fig.add_subplot(2, 3, 4)
    bin_map = results['bin_idx_map']
    im4 = ax4.imshow(bin_map, cmap='viridis')
    ax4.set_title(f"4. Radial Bin Assignment\n({results['num_bins']} discrete bins)", fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Bin Index')

    # Draw some concentric circles to show bin boundaries
    bin_edges = results['bin_edges']
    H, W = bin_map.shape
    for i in range(0, len(bin_edges), 8):  # Every 8th bin
        circle = Circle((cx, cy), bin_edges[i], fill=False,
                        color='white', linewidth=0.5, linestyle='--', alpha=0.7)
        ax4.add_patch(circle)

    # 5. Radial Energy Distribution
    ax5 = fig.add_subplot(2, 3, 5)
    radial_energy = results['radial_energy']
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax5.bar(bin_centers, radial_energy, width=(bin_edges[1] - bin_edges[0]) * 0.9,
            color='steelblue', edgecolor='navy', alpha=0.7)
    ax5.set_xlabel('Radial Distance (pixels)', fontsize=11)
    ax5.set_ylabel('Spectral Energy', fontsize=11)
    ax5.set_title("5. Radial Energy Distribution\nE(r) = sum of power in bin", fontsize=12, fontweight='bold')
    ax5.set_xlim(0, bin_edges[-1])
    ax5.grid(True, alpha=0.3)

    # Add annotation for peak
    peak_idx = np.argmax(radial_energy)
    peak_r = bin_centers[peak_idx]
    peak_e = radial_energy[peak_idx]
    ax5.annotate(f'Peak at r={peak_r:.1f}', xy=(peak_r, peak_e),
                xytext=(peak_r + 50, peak_e * 0.8),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    # 6. Normalized Distribution + RSE Calculation
    ax6 = fig.add_subplot(2, 3, 6)
    p = results['radial_distribution']

    ax6.bar(bin_centers, p, width=(bin_edges[1] - bin_edges[0]) * 0.9,
            color='darkorange', edgecolor='brown', alpha=0.7)
    ax6.set_xlabel('Radial Distance (pixels)', fontsize=11)
    ax6.set_ylabel('p(r) = E(r) / total', fontsize=11)
    ax6.set_title("6. Probability Distribution\nRSE = H(p) / log(K)", fontsize=12, fontweight='bold')
    ax6.set_xlim(0, bin_edges[-1])
    ax6.grid(True, alpha=0.3)

    # Add RSE calculation box
    rse = results['rse']
    raw_H = results['raw_entropy']
    max_H = results['max_entropy']

    textstr = (f"Shannon Entropy H = {raw_H:.4f}\n"
               f"Max Entropy = log({results['num_bins']}) = {max_H:.4f}\n"
               f"RSE = H / H_max = {rse:.4f}")

    props = dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.9)
    ax6.text(0.98, 0.98, textstr, transform=ax6.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Overall title
    fig.suptitle(f"Radial Spectral Entropy (RSE) Computation Pipeline\n"
                 f"Final RSE = {rse:.4f} (0=concentrated, 1=uniform)",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {output_path} (RSE = {rse:.4f})")

    return rse


def main():
    """Visualize RSE computation for sample images from each domain."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("RSE Computation Process Visualization")
    print("=" * 60)

    # Select sample images from each domain
    samples = []

    for domain in ["domain_A", "domain_B", "domain_C", "domain_D"]:
        domain_path = HDMAP_PNG_ROOT / domain

        # Get one normal and one abnormal sample
        normal_dir = domain_path / "train" / "good"
        abnormal_dir = domain_path / "test" / "fault"

        if normal_dir.exists():
            normal_files = sorted(normal_dir.glob("*.png"))
            if normal_files:
                samples.append((normal_files[0], f"{domain} - Normal"))

        if abnormal_dir.exists():
            abnormal_files = sorted(abnormal_dir.glob("*.png"))
            if abnormal_files:
                samples.append((abnormal_files[0], f"{domain} - Abnormal"))

    # Process each sample
    rse_values = []

    for img_path, label in samples:
        print(f"\nProcessing: {label}")
        safe_label = label.replace(" ", "_").replace("-", "_")
        output_path = OUTPUT_DIR / f"rse_process_{safe_label}.png"

        rse = visualize_rse_process(img_path, output_path, label)
        rse_values.append((label, rse))

    # Print summary
    print("\n" + "=" * 60)
    print("RSE Values Summary")
    print("=" * 60)
    print(f"{'Sample':<30} | {'RSE':>8}")
    print("-" * 42)
    for label, rse in rse_values:
        print(f"{label:<30} | {rse:>8.4f}")

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
