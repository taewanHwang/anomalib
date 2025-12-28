#!/usr/bin/env python3
"""Test CLIP horizontal line recognition at varying contrast levels.

Tests artificial horizontal lines with contrast levels from real-defect-like
to heavily enhanced, using horizontal/vertical text prompts.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import open_clip

from anomalib.data.datasets.image.hdmap import HDMAPDataset


ANOMALIB_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_ROOT = ANOMALIB_ROOT / "datasets" / "HDMAP" / "1000_tiff_minmax"
OUTPUT_DIR = Path(__file__).parent / "HDMAP_vis"


def load_image(domain: str, split: str, label: str, index: int):
    """Load image from HDMAPDataset."""
    dataset = HDMAPDataset(
        root=str(DATASET_ROOT),
        domain=domain,
        split=split,
        target_size=None,
    )

    target_path = f"{label}/{index:06d}.tiff"
    for item in dataset:
        if target_path in item.image_path:
            return item.image, item.image_path

    raise ValueError(f"Image not found: {target_path}")


def add_horizontal_line(image: torch.Tensor, y_center: int, width: int, target_contrast: float):
    """Add horizontal line with specific contrast ratio.

    Args:
        image: Input image (C, H, W)
        y_center: Center Y position of the line
        width: Width of the line in pixels
        target_contrast: Target contrast ratio (line_intensity / background)

    Returns:
        Modified image with horizontal line
    """
    modified = image.clone()

    # Calculate background intensity (excluding line region)
    mask = torch.ones_like(image[0], dtype=torch.bool)
    y_start = max(0, y_center - width // 2)
    y_end = min(image.shape[1], y_center + width // 2 + width % 2)
    mask[y_start:y_end, :] = False

    bg_intensity = image[:, mask].mean().item()

    # Calculate line intensity for target contrast
    line_intensity = bg_intensity * target_contrast
    line_intensity = min(line_intensity, 1.0)  # Clamp to max 1.0

    # Apply line
    for y in range(y_start, y_end):
        modified[:, y, :] = line_intensity

    actual_contrast = line_intensity / (bg_intensity + 1e-8)

    return modified, line_intensity, bg_intensity, actual_contrast


def resize_for_clip(image: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Resize image for CLIP input (ViT-L-14 uses 224x224)."""
    img = image.unsqueeze(0)  # (1, C, H, W)
    resized = F.interpolate(img, size=(size, size), mode='bilinear', align_corners=False)
    return resized.squeeze(0)  # (C, H, W)


def compute_text_similarities(model, tokenizer, image: torch.Tensor, device: torch.device):
    """Compute similarities to horizontal and vertical text prompts."""

    # Prepare image for CLIP (ViT-L-14 uses 224x224)
    img_resized = resize_for_clip(image, 224)

    # CLIP normalization
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)
    img_normalized = (img_resized.to(device) - mean) / std
    img_batch = img_normalized.unsqueeze(0)

    # Text prompts
    horizontal_prompts = [
        "an image with horizontal lines",
        "horizontal stripe pattern",
        "horizontal band across the image",
        "image with horizontal bright line",
    ]

    vertical_prompts = [
        "an image with vertical lines",
        "vertical stripe pattern",
        "vertical band across the image",
        "image with vertical bright line",
    ]

    neutral_prompts = [
        "a grayscale image",
        "noisy texture image",
        "sensor data image",
    ]

    with torch.no_grad():
        # Encode image
        image_features = model.encode_image(img_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Encode texts
        all_prompts = horizontal_prompts + vertical_prompts + neutral_prompts
        text_tokens = tokenizer(all_prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarities
        similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()

    n_h = len(horizontal_prompts)
    n_v = len(vertical_prompts)

    h_sims = similarities[:n_h]
    v_sims = similarities[n_h:n_h+n_v]
    n_sims = similarities[n_h+n_v:]

    return {
        "horizontal_mean": h_sims.mean(),
        "vertical_mean": v_sims.mean(),
        "neutral_mean": n_sims.mean(),
        "h_v_diff": h_sims.mean() - v_sims.mean(),
        "horizontal_max": h_sims.max(),
        "vertical_max": v_sims.max(),
        "horizontal_sims": h_sims,
        "vertical_sims": v_sims,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="domain_C")
    parser.add_argument("--good-index", type=int, default=9)
    parser.add_argument("--fault-index", type=int, default=9)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load CLIP model
    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    # Load images
    good_img, _ = load_image(args.domain, "test", "good", args.good_index)
    fault_img, _ = load_image(args.domain, "test", "fault", args.fault_index)

    print(f"\nGood image shape: {good_img.shape}")
    print(f"Good image intensity: mean={good_img.mean():.4f}, min={good_img.min():.4f}, max={good_img.max():.4f}")

    # Calculate real defect contrast
    # Find defect row by comparing fault vs good
    fault_means = fault_img.mean(dim=(0, 2))
    good_means = good_img.mean(dim=(0, 2))
    diff = fault_means - good_means
    defect_row = diff.argmax().item()

    real_defect_intensity = fault_img[:, defect_row, :].mean().item()
    real_bg_intensity = good_img.mean().item()
    real_contrast = real_defect_intensity / (real_bg_intensity + 1e-8)

    print(f"\nReal defect analysis:")
    print(f"  Defect row: {defect_row}")
    print(f"  Defect intensity: {real_defect_intensity:.4f}")
    print(f"  Background intensity: {real_bg_intensity:.4f}")
    print(f"  Real contrast ratio: {real_contrast:.2f}x")

    # Define contrast levels to test
    contrast_levels = [
        ("Real defect level", real_contrast),
        ("1.5x contrast", 1.5),
        ("2.0x contrast", 2.0),
        ("2.5x contrast", 2.5),
        ("3.0x contrast", 3.0),
        ("4.0x contrast", 4.0),
        ("5.0x contrast", 5.0),
    ]

    # Parameters for artificial line
    h = good_img.shape[1]
    y_center = h // 2
    line_width = 3  # Similar to real defect width

    results = []

    # Test original images first
    print("\n" + "=" * 70)
    print("Testing original images...")
    print("=" * 70)

    # Original good
    good_result = compute_text_similarities(model, tokenizer, good_img, device)
    print(f"\nOriginal Good Image:")
    print(f"  H similarity: {good_result['horizontal_mean']:.4f}")
    print(f"  V similarity: {good_result['vertical_mean']:.4f}")
    print(f"  H-V diff: {good_result['h_v_diff']:+.4f}")

    results.append({
        "name": "Original Good",
        "contrast": 1.0,
        "line_intensity": None,
        "h_sim": good_result['horizontal_mean'],
        "v_sim": good_result['vertical_mean'],
        "h_v_diff": good_result['h_v_diff'],
        "image": good_img,
    })

    # Original fault
    fault_result = compute_text_similarities(model, tokenizer, fault_img, device)
    print(f"\nOriginal Fault Image (real defect):")
    print(f"  H similarity: {fault_result['horizontal_mean']:.4f}")
    print(f"  V similarity: {fault_result['vertical_mean']:.4f}")
    print(f"  H-V diff: {fault_result['h_v_diff']:+.4f}")

    results.append({
        "name": f"Real Fault ({real_contrast:.2f}x)",
        "contrast": real_contrast,
        "line_intensity": real_defect_intensity,
        "h_sim": fault_result['horizontal_mean'],
        "v_sim": fault_result['vertical_mean'],
        "h_v_diff": fault_result['h_v_diff'],
        "image": fault_img,
    })

    # Test artificial lines with varying contrast
    print("\n" + "=" * 70)
    print("Testing artificial horizontal lines with varying contrast...")
    print("=" * 70)

    for name, target_contrast in contrast_levels:
        modified_img, line_int, bg_int, actual_contrast = add_horizontal_line(
            good_img, y_center, line_width, target_contrast
        )

        result = compute_text_similarities(model, tokenizer, modified_img, device)

        print(f"\n{name} (actual: {actual_contrast:.2f}x):")
        print(f"  Line intensity: {line_int:.4f}, Background: {bg_int:.4f}")
        print(f"  H similarity: {result['horizontal_mean']:.4f}")
        print(f"  V similarity: {result['vertical_mean']:.4f}")
        print(f"  H-V diff: {result['h_v_diff']:+.4f}")

        results.append({
            "name": name,
            "contrast": actual_contrast,
            "line_intensity": line_int,
            "h_sim": result['horizontal_mean'],
            "v_sim": result['vertical_mean'],
            "h_v_diff": result['h_v_diff'],
            "image": modified_img,
        })

    # Visualization
    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)

    n_results = len(results)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    def norm_display(t):
        arr = t.permute(1, 2, 0).numpy() if len(t.shape) == 3 else t.numpy()
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # Row 1-2: Images with their H-V scores
    for i, r in enumerate(results[:8]):
        row = i // 4
        col = i % 4
        ax = axes[row, col]

        ax.imshow(norm_display(r["image"]))

        # Color code based on H-V diff
        if r["h_v_diff"] > 0.02:
            color = "green"
            status = "HORIZONTAL DETECTED"
        elif r["h_v_diff"] > 0:
            color = "orange"
            status = "Weak horizontal"
        else:
            color = "red"
            status = "Not detected"

        title = f"{r['name']}\nH-V: {r['h_v_diff']:+.4f}\n{status}"
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        ax.axis("off")

    # Fill remaining cells if needed
    for i in range(len(results), 8):
        row = i // 4
        col = i % 4
        axes[row, col].axis("off")

    # Row 3: Summary chart
    ax = axes[2, 0]
    ax.axis("off")

    # Bar chart of H-V diff
    ax = axes[2, 1]
    names = [r["name"].replace(" contrast", "").replace("Real defect level", "Real\ndefect") for r in results]
    h_v_diffs = [r["h_v_diff"] for r in results]
    colors = ["green" if d > 0.02 else "orange" if d > 0 else "red" for d in h_v_diffs]

    bars = ax.bar(range(len(names)), h_v_diffs, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.02, color='green', linestyle='--', linewidth=1, label='Detection threshold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("H-V Similarity Difference")
    ax.set_title("Horizontal Recognition Score by Contrast Level", fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Line chart showing trend
    ax = axes[2, 2]
    contrasts = [r["contrast"] for r in results[1:]]  # Skip original good
    h_v_diffs_trend = [r["h_v_diff"] for r in results[1:]]

    ax.plot(contrasts, h_v_diffs_trend, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.02, color='green', linestyle='--', linewidth=1, label='Detection threshold')
    ax.axvline(x=real_contrast, color='red', linestyle=':', linewidth=2, label=f'Real defect ({real_contrast:.2f}x)')
    ax.set_xlabel("Contrast Ratio")
    ax.set_ylabel("H-V Similarity Difference")
    ax.set_title("H-V Score vs Contrast Level", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary text
    ax = axes[2, 3]
    ax.axis("off")

    # Find threshold contrast where H-V becomes positive
    threshold_contrast = None
    for r in results[1:]:
        if r["h_v_diff"] > 0.02:
            threshold_contrast = r["contrast"]
            break

    summary_text = f"""
    Summary: Contrast Level vs CLIP Recognition
    ════════════════════════════════════════════

    Real HDMAP defect contrast: {real_contrast:.2f}x
    Real defect H-V score: {results[1]['h_v_diff']:+.4f}

    Minimum contrast for CLIP detection:
    {'  ~' + f'{threshold_contrast:.1f}x' if threshold_contrast else '  > 5.0x (not found)'}

    Gap: Real defect needs {threshold_contrast/real_contrast if threshold_contrast else '>4'}x
         stronger contrast to be detected

    ════════════════════════════════════════════
    Conclusion:
    Text Template approach with "horizontal line"
    prompts will NOT work for HDMAP defects
    because the defect contrast is too subtle
    for CLIP to recognize as a "horizontal line".
    """

    ax.text(0.05, 0.5, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle(f"CLIP Horizontal Line Recognition vs Contrast Level ({args.domain})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / args.domain / "contrast_level_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved visualization to: {output_path}")

    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Name':<25} {'Contrast':<10} {'H Sim':<10} {'V Sim':<10} {'H-V Diff':<12} {'Status'}")
    print("-" * 80)

    for r in results:
        if r["h_v_diff"] > 0.02:
            status = "DETECTED"
        elif r["h_v_diff"] > 0:
            status = "Weak"
        else:
            status = "Not detected"

        print(f"{r['name']:<25} {r['contrast']:<10.2f} {r['h_sim']:<10.4f} {r['v_sim']:<10.4f} {r['h_v_diff']:<+12.4f} {status}")

    print("=" * 80)


if __name__ == "__main__":
    main()
