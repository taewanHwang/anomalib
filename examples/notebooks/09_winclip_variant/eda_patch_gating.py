"""EDA: Patch-level Gating Analysis for CA-WinCLIP.

This script analyzes whether patch-level median voting can improve
gating accuracy for fault/cold samples.

Hypothesis:
- Global embedding gating fails for fault/cold because fault patterns affect the global embedding
- Patch-level similarity should be more robust since faults are localized
- Median voting should ignore outlier patches affected by faults
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from anomalib.data.datasets.image.hdmap import HDMAPDataset
from anomalib.models.image.winclip.torch_model import WinClipModel


def load_samples(dataset_root: str, domain: str, indices: list, device: torch.device):
    """Load specific samples from dataset."""
    dataset = HDMAPDataset(
        root=dataset_root,
        domain=domain,
        split="test",
        target_size=(240, 240),
        resize_method="resize",
    )

    samples = []
    for idx in indices:
        sample = dataset[idx]
        samples.append({
            'image': sample.image.unsqueeze(0).to(device),
            'label': sample.gt_label,
            'path': sample.image_path,
            'index': idx,
        })
    return samples


def analyze_patch_similarities(model, test_img, ref_cold_img, ref_warm_img):
    """Analyze patch-level similarities between test and references.

    Returns:
        Dict with analysis results
    """
    with torch.no_grad():
        # Encode all images
        _, _, test_patches = model.encode_image(test_img)      # (1, 225, 896)
        _, _, cold_patches = model.encode_image(ref_cold_img)  # (1, 225, 896)
        _, _, warm_patches = model.encode_image(ref_warm_img)  # (1, 225, 896)

        # Get global embeddings too
        test_global, _, _ = model.encode_image(test_img)
        cold_global, _, _ = model.encode_image(ref_cold_img)
        warm_global, _, _ = model.encode_image(ref_warm_img)

    # Normalize embeddings
    test_patches_norm = torch.nn.functional.normalize(test_patches, dim=-1)
    cold_patches_norm = torch.nn.functional.normalize(cold_patches, dim=-1)
    warm_patches_norm = torch.nn.functional.normalize(warm_patches, dim=-1)

    test_global_norm = torch.nn.functional.normalize(test_global, dim=-1)
    cold_global_norm = torch.nn.functional.normalize(cold_global, dim=-1)
    warm_global_norm = torch.nn.functional.normalize(warm_global, dim=-1)

    # Compute patch-level similarities (each test patch vs corresponding ref patch)
    # Shape: (225,)
    sim_cold_patches = (test_patches_norm * cold_patches_norm).sum(dim=-1).squeeze()
    sim_warm_patches = (test_patches_norm * warm_patches_norm).sum(dim=-1).squeeze()

    # Compute global similarities
    sim_cold_global = (test_global_norm * cold_global_norm).sum(dim=-1).item()
    sim_warm_global = (test_global_norm * warm_global_norm).sum(dim=-1).item()

    # Base statistics
    global_selected = 'cold' if sim_cold_global > sim_warm_global else 'warm'
    patch_mean_selected = 'cold' if sim_cold_patches.mean() > sim_warm_patches.mean() else 'warm'
    patch_median_selected = 'cold' if sim_cold_patches.median() > sim_warm_patches.median() else 'warm'
    patch_top75_selected = 'cold' if sim_cold_patches.topk(int(0.75 * 225))[0].mean() > sim_warm_patches.topk(int(0.75 * 225))[0].mean() else 'warm'

    # Margins (confidence) for each method
    global_margin = abs(sim_cold_global - sim_warm_global)
    patch_mean_margin = abs(sim_cold_patches.mean().item() - sim_warm_patches.mean().item())
    patch_median_margin = abs(sim_cold_patches.median().item() - sim_warm_patches.median().item())

    # === Method 1: Ensemble/Voting (majority vote) ===
    votes = [global_selected, patch_mean_selected, patch_median_selected]
    cold_votes = votes.count('cold')
    warm_votes = votes.count('warm')
    ensemble_selected = 'cold' if cold_votes > warm_votes else 'warm'

    # === Method 2: Confidence-based (use method with highest margin) ===
    methods = {
        'global': (global_selected, global_margin),
        'patch_mean': (patch_mean_selected, patch_mean_margin),
        'patch_median': (patch_median_selected, patch_median_margin),
    }
    # Select method with highest margin
    best_method = max(methods.keys(), key=lambda m: methods[m][1])
    confidence_selected = methods[best_method][0]
    confidence_margin = methods[best_method][1]
    confidence_method = best_method

    results = {
        # Global embedding approach (current)
        'global_cold': sim_cold_global,
        'global_warm': sim_warm_global,
        'global_selected': global_selected,
        'global_margin': global_margin,

        # Patch-level mean (similar to global)
        'patch_mean_cold': sim_cold_patches.mean().item(),
        'patch_mean_warm': sim_warm_patches.mean().item(),
        'patch_mean_selected': patch_mean_selected,
        'patch_mean_margin': patch_mean_margin,

        # Patch-level median (more robust)
        'patch_median_cold': sim_cold_patches.median().item(),
        'patch_median_warm': sim_warm_patches.median().item(),
        'patch_median_selected': patch_median_selected,
        'patch_median_margin': patch_median_margin,

        # Patch-level top-75% mean (exclude outliers)
        'patch_top75_cold': sim_cold_patches.topk(int(0.75 * 225))[0].mean().item(),
        'patch_top75_warm': sim_warm_patches.topk(int(0.75 * 225))[0].mean().item(),
        'patch_top75_selected': patch_top75_selected,

        # === NEW: Ensemble (Majority Voting) ===
        'ensemble_selected': ensemble_selected,
        'ensemble_votes': {'cold': cold_votes, 'warm': warm_votes},

        # === NEW: Confidence-based ===
        'confidence_selected': confidence_selected,
        'confidence_method': confidence_method,
        'confidence_margin': confidence_margin,

        # Raw patch similarities for visualization
        'sim_cold_patches': sim_cold_patches.cpu().numpy(),
        'sim_warm_patches': sim_warm_patches.cpu().numpy(),
    }

    return results


def visualize_patch_similarities(results, test_info, test_image=None, save_path=None):
    """Visualize patch similarity distributions with test image."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    sim_cold = results['sim_cold_patches']
    sim_warm = results['sim_warm_patches']

    # 1. Test image
    ax = axes[0]
    if test_image is not None:
        # Convert from (C, H, W) to (H, W, C) for display
        img_np = test_image.cpu().squeeze(0).permute(1, 2, 0).numpy()
        # Normalize for display
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        ax.imshow(img_np)
        ax.set_title(f'Test Image\n{test_info}')
    else:
        ax.set_title('Test Image (N/A)')
    ax.axis('off')

    # 2. Histogram of patch similarities
    ax = axes[1]
    ax.hist(sim_cold, bins=30, alpha=0.6, label=f"vs Cold ref (median={np.median(sim_cold):.4f})")
    ax.hist(sim_warm, bins=30, alpha=0.6, label=f"vs Warm ref (median={np.median(sim_warm):.4f})")
    ax.axvline(np.median(sim_cold), color='blue', linestyle='--', alpha=0.8)
    ax.axvline(np.median(sim_warm), color='orange', linestyle='--', alpha=0.8)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title(f'Patch Similarity Distribution')
    ax.legend(fontsize=8)

    # 3. Spatial heatmap - Cold similarity
    ax = axes[2]
    sim_cold_2d = sim_cold.reshape(15, 15)  # 225 = 15x15
    im = ax.imshow(sim_cold_2d, cmap='coolwarm', vmin=0.3, vmax=1.0)
    ax.set_title(f'Similarity to Cold Ref\nmean={np.mean(sim_cold):.4f}')
    plt.colorbar(im, ax=ax)

    # 4. Spatial heatmap - Difference (Cold - Warm)
    ax = axes[3]
    diff = sim_cold - sim_warm
    diff_2d = diff.reshape(15, 15)
    im = ax.imshow(diff_2d, cmap='RdBu', vmin=-0.15, vmax=0.15)
    ax.set_title(f'Difference (Cold - Warm)\nmean={np.mean(diff):.4f}')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def evaluate_condition_group(model, samples, gt_condition, ref_cold, ref_warm, output_dir, group_name, save_viz=True):
    """Evaluate a group of samples and return results summary."""
    results_summary = {
        'global_correct': 0,
        'patch_mean_correct': 0,
        'patch_median_correct': 0,
        'patch_top75_correct': 0,
        'ensemble_correct': 0,
        'confidence_correct': 0,
        'total': 0,
    }

    all_results = []
    viz_count = 0

    for sample in samples:
        results = analyze_patch_similarities(
            model,
            sample['image'],
            ref_cold['image'],
            ref_warm['image']
        )
        results['gt'] = gt_condition
        results['index'] = sample['index']
        all_results.append(results)

        # Check correctness
        global_ok = "✓" if results['global_selected'] == gt_condition else "✗"
        mean_ok = "✓" if results['patch_mean_selected'] == gt_condition else "✗"
        median_ok = "✓" if results['patch_median_selected'] == gt_condition else "✗"
        top75_ok = "✓" if results['patch_top75_selected'] == gt_condition else "✗"
        ensemble_ok = "✓" if results['ensemble_selected'] == gt_condition else "✗"
        confidence_ok = "✓" if results['confidence_selected'] == gt_condition else "✗"

        results_summary['global_correct'] += 1 if results['global_selected'] == gt_condition else 0
        results_summary['patch_mean_correct'] += 1 if results['patch_mean_selected'] == gt_condition else 0
        results_summary['patch_median_correct'] += 1 if results['patch_median_selected'] == gt_condition else 0
        results_summary['patch_top75_correct'] += 1 if results['patch_top75_selected'] == gt_condition else 0
        results_summary['ensemble_correct'] += 1 if results['ensemble_selected'] == gt_condition else 0
        results_summary['confidence_correct'] += 1 if results['confidence_selected'] == gt_condition else 0
        results_summary['total'] += 1

        print(f"{sample['index']:<8} {gt_condition:<6} | "
              f"{results['global_selected']:<4} {global_ok:<4} | "
              f"{results['patch_mean_selected']:<6} {mean_ok:<4} | "
              f"{results['ensemble_selected']:<4} {ensemble_ok:<4} | "
              f"{results['confidence_selected']:<4} {confidence_ok:<4} ({results['confidence_method'][:6]})")

        # Save visualization for first 3 samples per group
        if save_viz and viz_count < 3:
            test_info = f"{group_name} idx={sample['index']}"
            visualize_patch_similarities(
                results,
                test_info,
                test_image=sample['image'],
                save_path=output_dir / f"patch_sim_{group_name.replace('/', '_')}_{sample['index']:04d}.png"
            )
            viz_count += 1

    return results_summary, all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--domain", type=str, default="domain_C")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples per condition group")
    args = parser.parse_args()

    # Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset_root = "/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/HDMAP/1000_tiff_minmax"
    output_dir = Path(__file__).parent / "eda_results" / "patch_gating"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Domain: {args.domain}")
    print(f"Output: {output_dir}")
    print(f"Samples per group: {args.num_samples}")

    # Load WinCLIP model
    print("\nLoading WinCLIP model...")
    model = WinClipModel()
    model.setup("industrial sensor data", None)
    model = model.to(device)
    model.eval()

    # Load reference images (good/cold and good/warm)
    # good/cold: dataset index 1000 (file index 0)
    # good/warm: dataset index 1999 (file index 999)
    print("\nLoading reference images...")
    refs = load_samples(dataset_root, args.domain, [1000, 1999], device)
    ref_cold = refs[0]  # good/cold
    ref_warm = refs[1]  # good/warm
    print(f"  Cold ref: {ref_cold['path']} (index {ref_cold['index']})")
    print(f"  Warm ref: {ref_warm['path']} (index {ref_warm['index']})")

    # Define all 4 condition groups
    # Dataset structure:
    #   0-499:    fault/cold (file 0-499)
    #   500-999:  fault/warm (file 500-999)
    #   1000-1499: good/cold (file 0-499)
    #   1500-1999: good/warm (file 500-999)
    #
    # Note: Reference images are at index 1000 (cold) and 1999 (warm)
    #       We need to exclude these from test samples
    ref_indices = {1000, 1999}  # Exclude reference images

    def get_indices(start, end, num_samples, exclude=None):
        """Get evenly spaced indices, excluding specific ones."""
        exclude = exclude or set()
        step = (end - start) // num_samples
        indices = []
        idx = start
        while len(indices) < num_samples and idx < end:
            if idx not in exclude:
                indices.append(idx)
            idx += step
        # If we need more samples, fill from the end
        idx = end - 1
        while len(indices) < num_samples and idx >= start:
            if idx not in exclude and idx not in indices:
                indices.append(idx)
            idx -= 1
        return sorted(indices)[:num_samples]

    condition_groups = {
        'fault/cold': {
            'indices': get_indices(0, 500, args.num_samples),
            'gt_condition': 'cold',
        },
        'fault/warm': {
            'indices': get_indices(500, 1000, args.num_samples),
            'gt_condition': 'warm',
        },
        'good/cold': {
            'indices': get_indices(1000, 1500, args.num_samples, exclude=ref_indices),
            'gt_condition': 'cold',
        },
        'good/warm': {
            'indices': get_indices(1500, 2000, args.num_samples, exclude=ref_indices),
            'gt_condition': 'warm',
        },
    }

    # Store results for all groups
    all_group_results = {}

    # Evaluate each group
    for group_name, group_config in condition_groups.items():
        print(f"\n{'='*100}")
        print(f"Testing {group_name.upper()} samples (GT = {group_config['gt_condition']})")
        print("=" * 100)
        print(f"{'Index':<8} {'GT':<6} | {'Global':<10} | {'Patch Mean':<12} | {'Ensemble':<10} | {'Confidence (method)':<20}")
        print("-" * 100)

        # Load samples
        samples = load_samples(dataset_root, args.domain, group_config['indices'], device)

        # Evaluate
        summary, results = evaluate_condition_group(
            model, samples, group_config['gt_condition'],
            ref_cold, ref_warm, output_dir, group_name
        )

        all_group_results[group_name] = summary

        # Print group summary
        print("-" * 100)
        total = summary['total']
        print(f"  Global: {summary['global_correct']}/{total} ({100*summary['global_correct']/total:.1f}%) | "
              f"Mean: {summary['patch_mean_correct']}/{total} ({100*summary['patch_mean_correct']/total:.1f}%) | "
              f"Ensemble: {summary['ensemble_correct']}/{total} ({100*summary['ensemble_correct']/total:.1f}%) | "
              f"Confidence: {summary['confidence_correct']}/{total} ({100*summary['confidence_correct']/total:.1f}%)")

    # Final comparison summary
    print("\n" + "=" * 100)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 100)
    print(f"\n{'Group':<15} | {'Global':<10} | {'P.Mean':<10} | {'P.Median':<10} | {'Ensemble':<10} | {'Confidence':<10}")
    print("-" * 90)

    totals = {'global': 0, 'mean': 0, 'median': 0, 'top75': 0, 'ensemble': 0, 'confidence': 0, 'total': 0}

    for group_name, summary in all_group_results.items():
        total = summary['total']
        g_pct = 100 * summary['global_correct'] / total
        m_pct = 100 * summary['patch_mean_correct'] / total
        md_pct = 100 * summary['patch_median_correct'] / total
        ens_pct = 100 * summary['ensemble_correct'] / total
        conf_pct = 100 * summary['confidence_correct'] / total

        print(f"{group_name:<15} | {g_pct:>8.1f}% | {m_pct:>8.1f}% | {md_pct:>8.1f}% | {ens_pct:>8.1f}% | {conf_pct:>8.1f}%")

        totals['global'] += summary['global_correct']
        totals['mean'] += summary['patch_mean_correct']
        totals['median'] += summary['patch_median_correct']
        totals['ensemble'] += summary['ensemble_correct']
        totals['confidence'] += summary['confidence_correct']
        totals['total'] += total

    # Overall accuracy
    print("-" * 90)
    g_total = 100 * totals['global'] / totals['total']
    m_total = 100 * totals['mean'] / totals['total']
    md_total = 100 * totals['median'] / totals['total']
    ens_total = 100 * totals['ensemble'] / totals['total']
    conf_total = 100 * totals['confidence'] / totals['total']
    print(f"{'OVERALL':<15} | {g_total:>8.1f}% | {m_total:>8.1f}% | {md_total:>8.1f}% | {ens_total:>8.1f}% | {conf_total:>8.1f}%")

    # Highlight key findings
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)

    # Find problematic groups and improvements
    print("\nPer-group Analysis:")
    for group_name, summary in all_group_results.items():
        total = summary['total']
        g_pct = 100 * summary['global_correct'] / total
        ens_pct = 100 * summary['ensemble_correct'] / total
        conf_pct = 100 * summary['confidence_correct'] / total

        best_method = max([
            ('Global', g_pct),
            ('Ensemble', ens_pct),
            ('Confidence', conf_pct),
        ], key=lambda x: x[1])

        status = "⚠️" if g_pct < 80 else "✓"
        print(f"  {status} {group_name}: Global={g_pct:.1f}%, Ensemble={ens_pct:.1f}%, Confidence={conf_pct:.1f}% -> Best: {best_method[0]} ({best_method[1]:.1f}%)")

    # Overall comparison
    print("\nOverall Comparison:")
    print(f"  - Global:     {g_total:.1f}%")
    print(f"  - Ensemble:   {ens_total:.1f}% ({'+' if ens_total > g_total else ''}{ens_total - g_total:.1f}%)")
    print(f"  - Confidence: {conf_total:.1f}% ({'+' if conf_total > g_total else ''}{conf_total - g_total:.1f}%)")

    best_overall = max([
        ('Global', g_total),
        ('Ensemble', ens_total),
        ('Confidence', conf_total),
    ], key=lambda x: x[1])
    print(f"\n  ★ Best Overall: {best_overall[0]} with {best_overall[1]:.1f}%")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
