"""
Radial Spectral Entropy (RSE) Analysis for HDMAP Dataset.

RSE measures how energy is distributed across radial frequency bands in the 2D FFT.
- Low RSE: Energy concentrated (regular/periodic patterns)
- High RSE: Energy spread (complex/irregular patterns)

This is more "representation-agnostic" than Orientation Entropy,
working well for HDMap, GAF, Recurrence maps, Spectrograms, etc.
"""

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

# Paths
HDMAP_PNG_ROOT = Path("/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_png")
OUTPUT_DIR = Path(__file__).parent / "results" / "rse_analysis"

DOMAINS = ["domain_A", "domain_B", "domain_C", "domain_D"]

# Dinomaly preprocessing (same as model input)
DINOMALY_TRANSFORM = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_rse_batch(
    images: torch.Tensor,
    num_radial_bins: int = 64,
    apply_hann_window: bool = True,
    remove_dc: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute Radial Spectral Entropy (RSE) for a batch of images (GPU-friendly).

    Parameters
    ----------
    images : torch.Tensor
        Batch of images (B, C, H, W), can be ImageNet normalized.
    num_radial_bins : int
        Number of radial bins for energy aggregation.
    apply_hann_window : bool
        Apply 2D Hann window before FFT to reduce spectral leakage.
    remove_dc : bool
        Remove DC component by subtracting mean.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    torch.Tensor
        RSE values of shape (B,) in [0, 1].
    """
    B, C, H, W = images.shape
    device = images.device

    # Convert to grayscale if RGB
    if C == 3:
        # Denormalize from ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        images_denorm = images * std + mean

        # ITU-R BT.601 luma
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
        gray = (images_denorm * weights).sum(dim=1)  # (B, H, W)
    else:
        gray = images.squeeze(1)  # (B, H, W)

    # Remove DC if requested
    if remove_dc:
        gray = gray - gray.mean(dim=(1, 2), keepdim=True)

    # Apply Hann window if requested
    if apply_hann_window:
        wy = torch.hann_window(H, device=device).view(H, 1)
        wx = torch.hann_window(W, device=device).view(1, W)
        window = wy * wx  # (H, W)
        gray = gray * window.unsqueeze(0)

    # 2D FFT power spectrum
    F_complex = torch.fft.fft2(gray)
    P = torch.abs(F_complex) ** 2
    P = torch.fft.fftshift(P, dim=(-2, -1))

    # Build radial distance map from center
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    yy = torch.arange(H, device=device, dtype=torch.float32)
    xx = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)  # (H, W)

    r_max = rr.max().item()
    if r_max <= 0:
        return torch.zeros(B, device=device)

    # Radial binning
    bin_edges = torch.linspace(0.0, r_max + eps, num_radial_bins + 1, device=device)
    bin_idx = torch.bucketize(rr.flatten(), bin_edges) - 1
    bin_idx = bin_idx.clamp(0, num_radial_bins - 1)  # (H*W,)

    # Compute RSE for each image in batch
    rse_values = torch.zeros(B, device=device)

    for i in range(B):
        P_flat = P[i].flatten()  # (H*W,)

        # Sum energy per radial bin
        radial_energy = torch.zeros(num_radial_bins, device=device)
        radial_energy.scatter_add_(0, bin_idx, P_flat)

        total = radial_energy.sum()
        if total <= 0:
            rse_values[i] = 0.0
            continue

        p = radial_energy / total  # radial energy distribution

        # Shannon entropy (natural log), normalized by log(K)
        H_r = -torch.sum(p * torch.log(p + eps))
        H_norm = H_r / np.log(num_radial_bins)

        rse_values[i] = H_norm.clamp(0.0, 1.0)

    return rse_values


def compute_rse_for_folder(folder: Path, batch_size: int = 64, device: str = "cuda") -> list[float]:
    """Compute RSE for all images in a folder using Dinomaly preprocessing."""
    rse_values = []

    if not folder.exists():
        print(f"Warning: {folder} does not exist")
        return rse_values

    image_files = sorted(folder.glob("*.png"))

    # Process in batches
    for i in tqdm(range(0, len(image_files), batch_size), desc=f"  {folder.name}", leave=False):
        batch_files = image_files[i:i+batch_size]

        # Load and transform images
        batch_images = []
        for img_path in batch_files:
            img = Image.open(img_path).convert("RGB")
            img_tensor = DINOMALY_TRANSFORM(img)
            batch_images.append(img_tensor)

        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)

            # Compute RSE
            batch_rse = compute_rse_batch(batch_tensor)
            rse_values.extend(batch_rse.cpu().numpy().tolist())

    return rse_values


def analyze_domain(domain: str, device: str = "cuda") -> pd.DataFrame:
    """Analyze RSE for a single domain."""
    domain_path = HDMAP_PNG_ROOT / domain

    data = []

    # Normal Train
    print(f"  Processing train/good...")
    train_good_rse = compute_rse_for_folder(domain_path / "train" / "good", device=device)
    for rse in train_good_rse:
        data.append({"category": "Normal Train", "rse": rse})

    # Normal Test
    print(f"  Processing test/good...")
    test_good_rse = compute_rse_for_folder(domain_path / "test" / "good", device=device)
    for rse in test_good_rse:
        data.append({"category": "Normal Test", "rse": rse})

    # Abnormal Test
    print(f"  Processing test/fault...")
    test_fault_rse = compute_rse_for_folder(domain_path / "test" / "fault", device=device)
    for rse in test_fault_rse:
        data.append({"category": "Abnormal Test", "rse": rse})

    return pd.DataFrame(data)


def create_violin_plot(df: pd.DataFrame, domain: str, output_path: Path):
    """Create violin plot for a domain."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"Normal Train": "#90EE90", "Normal Test": "#87CEEB", "Abnormal Test": "#F08080"}
    order = ["Normal Train", "Normal Test", "Abnormal Test"]

    sns.violinplot(
        data=df,
        x="category",
        y="rse",
        order=order,
        palette=colors,
        ax=ax,
    )

    # Add mean annotations
    for i, cat in enumerate(order):
        cat_data = df[df["category"] == cat]["rse"]
        mean_val = cat_data.mean()
        ax.annotate(
            f"{mean_val:.3f}",
            xy=(i, mean_val),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=10,
            ha="left",
        )
        ax.hlines(mean_val, i - 0.3, i + 0.3, colors="black", linewidth=2)

    ax.set_title(f"{domain}", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Radial Spectral Entropy (RSE)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_combined_plot(all_data: dict[str, pd.DataFrame], output_path: Path):
    """Create combined violin plot for all domains."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {"Normal Train": "#90EE90", "Normal Test": "#87CEEB", "Abnormal Test": "#F08080"}
    order = ["Normal Train", "Normal Test", "Abnormal Test"]

    for ax, domain in zip(axes.flat, DOMAINS):
        df = all_data[domain]

        sns.violinplot(
            data=df,
            x="category",
            y="rse",
            order=order,
            palette=colors,
            ax=ax,
        )

        # Add mean annotations
        for i, cat in enumerate(order):
            cat_data = df[df["category"] == cat]["rse"]
            mean_val = cat_data.mean()
            ax.annotate(
                f"{mean_val:.3f}",
                xy=(i, mean_val),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=9,
                ha="left",
            )
            ax.hlines(mean_val, i - 0.3, i + 0.3, colors="black", linewidth=2)

        ax.set_title(f"{domain}", fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("RSE")

    fig.suptitle(
        "Radial Spectral Entropy Distribution by Domain (HDMAP PNG)\n"
        "Dinomaly preprocessing: 448x448 resize + ImageNet normalize",
        fontsize=13,
        fontweight="bold",
    )

    # Add legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[cat]) for cat in order]
    fig.legend(handles, order, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.97))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined figure: {output_path}")


def print_summary(all_data: dict[str, pd.DataFrame]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY: Radial Spectral Entropy Statistics (Dinomaly preprocessing)")
    print("=" * 80)
    print(f"\n{'Domain':<12} | {'Category':<15} | {'Count':>6} | {'Mean':>8} | {'Std':>8} | {'Median':>8}")
    print("-" * 80)

    overall_normal = []
    overall_abnormal = []

    for domain in DOMAINS:
        df = all_data[domain]

        for cat in ["Normal Train", "Normal Test", "Abnormal Test"]:
            cat_data = df[df["category"] == cat]["rse"]
            print(f"{domain:<12} | {cat:<15} | {len(cat_data):>6} | {cat_data.mean():>8.4f} | {cat_data.std():>8.4f} | {cat_data.median():>8.4f}")

            if cat in ["Normal Train", "Normal Test"]:
                overall_normal.extend(cat_data.tolist())
            else:
                overall_abnormal.extend(cat_data.tolist())

        print("-" * 80)

    # Overall comparison
    print("\n" + "=" * 80)
    print("COMPARISON: Normal (Train+Test) vs Abnormal")
    print("=" * 80)

    print(f"\n{'Domain':<12} | {'Normal Mean':>12} | {'Abnormal Mean':>14} | {'Difference':>12}")
    print("-" * 80)

    all_normal_mean = []
    all_abnormal_mean = []

    for domain in DOMAINS:
        df = all_data[domain]
        normal_data = df[df["category"].isin(["Normal Train", "Normal Test"])]["rse"]
        abnormal_data = df[df["category"] == "Abnormal Test"]["rse"]

        normal_mean = normal_data.mean()
        abnormal_mean = abnormal_data.mean()
        diff = abnormal_mean - normal_mean

        all_normal_mean.append(normal_mean)
        all_abnormal_mean.append(abnormal_mean)

        print(f"{domain:<12} | {normal_mean:>12.4f} | {abnormal_mean:>14.4f} | {diff:>+12.4f}")

    print("-" * 80)

    overall_normal_mean = np.mean(all_normal_mean)
    overall_abnormal_mean = np.mean(all_abnormal_mean)
    overall_diff = overall_abnormal_mean - overall_normal_mean

    print(f"{'OVERALL':<12} | {overall_normal_mean:>12.4f} | {overall_abnormal_mean:>14.4f} | {overall_diff:>+12.4f}")
    print("=" * 80)

    print(f"\n>>> RECOMMENDED normal_rse for adaptive dropout: {overall_normal_mean:.4f}")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    if overall_diff > 0:
        print("RSE(Abnormal) > RSE(Normal): Anomalies have more spread frequency energy")
        print("  → For dropout: RSE < normal_rse (more regular) → higher dropout")
        print("  → This is OPPOSITE to Orientation Entropy direction!")
    else:
        print("RSE(Abnormal) < RSE(Normal): Anomalies have more concentrated frequency energy")
        print("  → For dropout: RSE > normal_rse (more regular) → higher dropout")


def main():
    """Main function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_data = {}

    # Analyze each domain
    for domain in DOMAINS:
        print(f"\n{'='*60}")
        print(f"Analyzing {domain}")
        print("=" * 60)

        df = analyze_domain(domain, device=device)
        all_data[domain] = df

        # Save individual violin plot
        create_violin_plot(df, domain, OUTPUT_DIR / f"rse_violin_{domain}.png")

        # Save data
        df.to_csv(OUTPUT_DIR / f"rse_data_{domain}.csv", index=False)

    # Create combined plot
    create_combined_plot(all_data, OUTPUT_DIR / "rse_violin_all_domains.png")

    # Print summary
    print_summary(all_data)

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
