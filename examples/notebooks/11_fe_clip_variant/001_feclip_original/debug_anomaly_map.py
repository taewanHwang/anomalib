"""Debug script to visualize anomaly map pipeline."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.nn import functional as F

from anomalib.data import MVTecAD
from anomalib.models.image import FEClip
from torchvision.transforms.v2 import Resize, CenterCrop, InterpolationMode

# Settings
DEVICE = "cuda:0"
CATEGORY = "bottle"
SAMPLE_IDX = 5  # Select an anomaly sample

def main():
    print("Loading trained model...")
    # Load trained checkpoint
    checkpoint_path = "examples/notebooks/11_fe_clip_variant/results/feclip_mvtec_seed42_20260101_061222/checkpoints/best_model.pt"

    model = FEClip(
        tap_indices=[20, 21, 22, 23],  # last4
    )
    model.to(DEVICE)
    model.model.setup_text()  # Initialize text embeddings

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")

    model.eval()

    # Load a test sample
    dm = MVTecAD(root="/mnt/ex-disk/taewan.hwang/study/anomalib/datasets/MVTecAD", category=CATEGORY, eval_batch_size=1)
    dm.setup()

    # Find an anomaly sample
    for i, batch in enumerate(dm.test_dataloader()):
        if batch.gt_label[0].item() == 1:  # anomaly
            if i >= SAMPLE_IDX:
                break

    print(f"Sample: {batch.image_path[0]}")
    print(f"GT Label: {batch.gt_label[0].item()}")

    # Preprocess image
    transform = model.pre_processor.transform
    images = transform(batch.image).to(DEVICE)
    print(f"Input image shape: {images.shape}")

    # Get per-tap maps
    with torch.no_grad():
        scores, map_list = model.model.forward_tokens(images)

    print(f"Number of taps: {len(map_list)}")
    print(f"Per-tap map shape: {map_list[0].shape}")  # Should be (1, 24, 24)

    # Average map
    avg_map = torch.stack(map_list, dim=0).mean(dim=0)  # (1, 24, 24)
    print(f"Averaged map shape: {avg_map.shape}")

    # Upsample to 336x336
    upsampled_map = F.interpolate(
        avg_map.unsqueeze(1),
        size=(336, 336),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    print(f"Upsampled map shape: {upsampled_map.shape}")

    # GT mask preprocessing (same as in evaluation)
    mask_resize = Resize(336, interpolation=InterpolationMode.NEAREST, antialias=False)
    mask_crop = CenterCrop((336, 336))

    gt_mask = batch.gt_mask[0]
    if gt_mask.ndim == 2:
        gt_mask = gt_mask.unsqueeze(0)
    gt_mask_processed = mask_crop(mask_resize(gt_mask)).squeeze(0)
    print(f"GT mask shape: {gt_mask_processed.shape}")

    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Per-tap maps (24x24)
    for i, tap_map in enumerate(map_list):
        ax = axes[0, i]
        im = ax.imshow(tap_map[0].cpu().numpy(), cmap='jet')
        ax.set_title(f"Tap {i} (24×24)")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: Pipeline stages
    # Input image
    img_display = images[0].cpu().permute(1, 2, 0).numpy()
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    axes[1, 0].imshow(img_display)
    axes[1, 0].set_title("Input (336×336)")
    axes[1, 0].axis('off')

    # Averaged map (24x24)
    im1 = axes[1, 1].imshow(avg_map[0].cpu().numpy(), cmap='jet')
    axes[1, 1].set_title("Avg Map (24×24)")
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)

    # Upsampled map (336x336)
    im2 = axes[1, 2].imshow(upsampled_map[0].cpu().numpy(), cmap='jet')
    axes[1, 2].set_title("Upsampled (336×336)")
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)

    # GT mask
    axes[1, 3].imshow(gt_mask_processed.cpu().numpy(), cmap='gray')
    axes[1, 3].set_title("GT Mask (336×336)")
    axes[1, 3].axis('off')

    plt.tight_layout()
    save_path = Path("examples/notebooks/11_fe_clip_variant/debug_anomaly_map.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")

    # Print statistics
    print("\n=== Statistics ===")
    print(f"Per-tap map ranges:")
    for i, tap_map in enumerate(map_list):
        print(f"  Tap {i}: min={tap_map.min():.4f}, max={tap_map.max():.4f}, mean={tap_map.mean():.4f}")
    print(f"Averaged map: min={avg_map.min():.4f}, max={avg_map.max():.4f}, mean={avg_map.mean():.4f}")
    print(f"Upsampled map: min={upsampled_map.min():.4f}, max={upsampled_map.max():.4f}")
    print(f"GT mask: unique values = {torch.unique(gt_mask_processed)}")

if __name__ == "__main__":
    main()
