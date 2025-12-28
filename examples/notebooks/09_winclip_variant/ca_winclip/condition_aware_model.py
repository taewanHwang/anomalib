"""Condition-Aware WinCLIP wrapper for N-Bank reference selection.

This module provides the ConditionAwareWinCLIP wrapper that extends WinCLIP
with condition-aware bank selection. The wrapper maintains separate reference
banks (e.g., cold/warm) and automatically selects the appropriate bank for
each test image using global embedding similarity.

The architecture follows the WinCLIP few-shot scoring pattern but adds:
1. Multiple reference banks instead of a single reference set
2. Gating mechanism to select the appropriate bank per test image
3. Bank-specific scoring using only the selected bank's embeddings

Supported gating mechanisms:
- MultiConditionGating: CLIP embedding-based (88.8% accuracy)
- P90IntensityGating: Intensity-based p90 percentile (96.7% accuracy, recommended)
- OracleGating: Ground truth-based (100%, for evaluation only)

Example:
    >>> from ca_winclip import ConditionAwareWinCLIP, P90IntensityGating
    >>> base_model = WinClipModel()
    >>> base_model.setup("industrial sensor data", None)
    >>> reference_banks = {"cold": cold_refs, "warm": warm_refs}
    >>> # Use P90 gating (recommended)
    >>> gating = P90IntensityGating(domain="domain_C")
    >>> ca_model = ConditionAwareWinCLIP(base_model, reference_banks, gating=gating)
    >>> scores, banks, details = ca_model.forward(test_batch, images=test_images)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from anomalib.models.image.winclip.utils import harmonic_aggregation, visual_association_score

from .gating import InverseGating, MultiConditionGating, OracleGating, P90IntensityGating, RandomGating

logger = logging.getLogger(__name__)


@dataclass
class BankEmbeddings:
    """Container for a single bank's embeddings.

    Stores all embedding types needed for WinCLIP scoring:
    - image_embeddings: Global embeddings used for gating
    - visual_embeddings: Multi-scale window embeddings for scoring
    - patch_embeddings: Full-resolution patch embeddings for scoring

    Attributes:
        image_embeddings: Global image embeddings, shape (N_refs, D).
        visual_embeddings: List of window embeddings per scale, each (N_refs, W, D).
        patch_embeddings: Patch embeddings, shape (N_refs, P, D).
    """

    image_embeddings: torch.Tensor  # (N_refs, D) - for gating
    visual_embeddings: List[torch.Tensor]  # [(N_refs, W, D), ...] - for scoring
    patch_embeddings: torch.Tensor  # (N_refs, P, D) - for scoring


class ConditionAwareWinCLIP(nn.Module):
    """N-Bank Condition-Aware WinCLIP wrapper.

    This wrapper extends WinCLIP with condition-aware bank selection. Instead of
    using a single reference bank, it maintains N banks (e.g., cold/warm) and
    automatically selects the most appropriate bank for each test image.

    The gating mechanism compares the test image's global embedding with each
    bank's reference embeddings and selects the bank with highest similarity.
    Anomaly scoring is then performed using only the selected bank.

    Attributes:
        base_model: The underlying WinClipModel.
        banks: Dictionary mapping bank names to BankEmbeddings.
        gating: The gating mechanism for bank selection.
        oracle_gating: Oracle gating for evaluation (optional).
        use_oracle: Whether to use oracle gating instead of learned gating.

    Example:
        >>> base_model = WinClipModel()
        >>> base_model.setup("industrial sensor data", None)
        >>> reference_banks = {"cold": cold_refs, "warm": warm_refs}
        >>> ca_model = ConditionAwareWinCLIP(base_model, reference_banks, gating_k=1)
        >>> scores, selected_banks, gating_details = ca_model.forward(test_batch)
    """

    def __init__(
        self,
        base_model,
        reference_banks: Dict[str, torch.Tensor],
        gating: Optional[Union[MultiConditionGating, P90IntensityGating]] = None,
        gating_k: int = 1,
        use_oracle: bool = False,
        verbose_gating: bool = False,
    ):
        """Initialize the Condition-Aware WinCLIP wrapper.

        Args:
            base_model: A configured WinClipModel instance with text embeddings set.
            reference_banks: Dictionary mapping bank names to reference image tensors.
                Each tensor should have shape (N_refs, 3, H, W).
            gating: Optional pre-configured gating instance. If provided, this will
                be used instead of creating a new MultiConditionGating.
                Recommended: P90IntensityGating for HDMAP dataset (96.7% accuracy).
            gating_k: Number of top references to average for CLIP-based gating.
                Ignored if `gating` parameter is provided.
            use_oracle: If True, use oracle gating based on index instead of
                learned/configured gating.
            verbose_gating: If True, log detailed gating decisions.
                Ignored if `gating` parameter is provided.
        """
        super().__init__()
        self.base_model = base_model
        self.banks: Dict[str, BankEmbeddings] = {}
        self.use_oracle = use_oracle
        self.use_p90_gating = isinstance(gating, P90IntensityGating)
        self.use_random_gating = isinstance(gating, RandomGating)
        self.use_inverse_gating = isinstance(gating, InverseGating)

        gating_type = type(gating).__name__ if gating else "MultiConditionGating"
        logger.info(f"Initializing CA-WinCLIP with {len(reference_banks)} banks, gating={gating_type}, use_oracle={use_oracle}")

        # Extract and cache embeddings for each bank
        gating_embeddings = {}
        with torch.no_grad():
            for bank_name, refs in reference_banks.items():
                # Move references to same device as model
                device = next(base_model.parameters()).device
                refs = refs.to(device)

                logger.info(f"  Encoding {bank_name} bank: {refs.shape[0]} references...")

                # Extract embeddings using base model
                img_emb, visual_emb, patch_emb = base_model.encode_image(refs)

                # Store bank embeddings
                self.banks[bank_name] = BankEmbeddings(
                    image_embeddings=img_emb,
                    visual_embeddings=visual_emb,
                    patch_embeddings=patch_emb,
                )
                gating_embeddings[bank_name] = img_emb

                logger.info(f"    -> image_emb: {img_emb.shape}, patch_emb: {patch_emb.shape}")

        # Initialize gating mechanisms
        if gating is not None:
            self.gating = gating
            logger.info(f"  Using provided gating: {type(gating).__name__}")
            if isinstance(gating, P90IntensityGating):
                logger.info(f"    P90 threshold: {gating.threshold:.4f}")
        else:
            self.gating = MultiConditionGating(gating_embeddings, k=gating_k, verbose=verbose_gating)
            logger.info(f"  Using CLIP-based gating with k={gating_k}")

        self.oracle_gating = OracleGating()

        logger.info(f"CA-WinCLIP initialized successfully. Banks: {list(self.banks.keys())}")

    @torch.no_grad()
    def forward(
        self,
        batch: torch.Tensor,
        indices: List[int] = None,
        images: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[str], List[Dict]]:
        """Forward pass with condition-aware bank selection.

        Uses bank-wise batching for efficient GPU utilization:
        1. Extract embeddings for all images (batched)
        2. Perform gating to determine bank selection for each image
        3. Group images by selected bank
        4. Compute scores for each bank group in batch (2 GPU calls instead of B)
        5. Reassemble results in original order

        Args:
            batch: Input images of shape (B, 3, H, W).
            indices: Optional list of sample indices for oracle gating and
                ground truth evaluation.
            images: Optional list of raw image tensors for P90 gating.
                Required when using P90IntensityGating. Each tensor should be
                shape (3, H, W) or (H, W, 3).

        Returns:
            Tuple containing:
            - scores: Anomaly scores of shape (B, H, W).
            - selected_banks: List of selected bank names for each image.
            - gating_details: List of dicts with similarity scores for each bank.
        """
        # Encode test images (batched - efficient)
        img_emb, window_emb, patch_emb = self.base_model.encode_image(batch)

        batch_size = len(img_emb)
        selected_banks = []
        gating_details = []

        # Step 1: Perform gating for all samples (lightweight CPU operations)
        for i in range(batch_size):
            if self.use_oracle and indices is not None:
                selected = self.oracle_gating.select_bank_by_index(indices[i])
                bank_scores = {name: 0.0 for name in self.banks.keys()}
            elif self.use_random_gating:
                file_idx = indices[i] if indices is not None else None
                selected, bank_scores = self.gating.select_bank(None, file_index=file_idx)
            elif self.use_inverse_gating:
                file_idx = indices[i] if indices is not None else None
                selected, bank_scores = self.gating.select_bank(None, file_index=file_idx)
            elif self.use_p90_gating:
                if images is not None:
                    image = images[i]
                else:
                    image = batch[i]
                file_idx = indices[i] if indices is not None else None
                selected, bank_scores = self.gating.select_bank(image, file_index=file_idx)
            else:
                file_idx = indices[i] if indices is not None else None
                selected, bank_scores = self.gating.select_bank(img_emb[i], file_index=file_idx)

            selected_banks.append(selected)
            gating_details.append(bank_scores)

        # Step 2: Group samples by selected bank
        bank_indices = {bank_name: [] for bank_name in self.banks.keys()}
        for i, bank_name in enumerate(selected_banks):
            bank_indices[bank_name].append(i)

        # Step 3: Compute scores for each bank group in batch (efficient GPU ops)
        # Pre-allocate score tensor
        grid_size = self.base_model.grid_size
        scores = torch.zeros(batch_size, *grid_size, device=batch.device)

        for bank_name, sample_indices in bank_indices.items():
            if not sample_indices:
                continue

            bank = self.banks[bank_name]

            # Extract embeddings for this bank's samples
            bank_patch_emb = patch_emb[sample_indices]  # (N, P, D)
            bank_window_emb = [we[sample_indices] for we in window_emb]  # [(N, W, D), ...]

            # Compute scores in batch for this bank
            bank_scores = self._compute_scores_with_bank_batch(
                bank_patch_emb, bank_window_emb, bank
            )

            # Place scores back in original order
            for local_idx, global_idx in enumerate(sample_indices):
                scores[global_idx] = bank_scores[local_idx]

        # Resize to original image dimensions
        pixel_scores = nn.functional.interpolate(
            scores.unsqueeze(1),
            size=batch.shape[-2:],
            mode="bilinear",
        ).squeeze(1)

        return pixel_scores, selected_banks, gating_details

    def _compute_scores_with_bank_batch(
        self,
        patch_embeddings: torch.Tensor,
        window_embeddings: List[torch.Tensor],
        bank: BankEmbeddings,
    ) -> torch.Tensor:
        """Compute multi-scale anomaly scores for a batch using a specific bank.

        This is the batched version of _compute_score_with_bank for efficient
        GPU utilization when multiple samples use the same bank.

        Args:
            patch_embeddings: Patch embeddings for batch, shape (N, P, D).
            window_embeddings: List of window embeddings, each (N, W, D).
            bank: BankEmbeddings containing reference embeddings.

        Returns:
            Anomaly score maps of shape (N, H, W).
        """
        grid_size = self.base_model.grid_size
        batch_size = patch_embeddings.shape[0]

        # Patch-level visual association scores (batched)
        # visual_association_score handles batch dimension
        patch_scores = visual_association_score(
            patch_embeddings, bank.patch_embeddings
        ).reshape(batch_size, *grid_size)

        multi_scale_scores = [patch_scores]

        # Window-level visual association scores for each scale (batched)
        for window_emb, ref_emb, mask in zip(
            window_embeddings,
            bank.visual_embeddings,
            self.base_model.masks,
            strict=True,
        ):
            scores = visual_association_score(window_emb, ref_emb)
            # harmonic_aggregation expects (N, W) and returns (N, H, W)
            agg_scores = harmonic_aggregation(scores, grid_size, mask)
            multi_scale_scores.append(agg_scores)

        # Average across scales: (num_scales, N, H, W) -> (N, H, W)
        return torch.stack(multi_scale_scores).mean(dim=0)

    def _compute_score_with_bank(
        self,
        patch_embeddings: torch.Tensor,
        window_embeddings: List[torch.Tensor],
        bank: BankEmbeddings,
    ) -> torch.Tensor:
        """Compute multi-scale anomaly scores using a specific bank.

        This method mirrors WinCLIP's _compute_few_shot_scores but uses
        the specified bank's embeddings instead of the global reference.

        Args:
            patch_embeddings: Patch embeddings for single image, shape (1, P, D).
            window_embeddings: List of window embeddings, each (1, W, D).
            bank: BankEmbeddings containing reference embeddings.

        Returns:
            Anomaly score map of shape (1, H, W).
        """
        grid_size = self.base_model.grid_size

        # Patch-level visual association score
        multi_scale_scores = [
            visual_association_score(patch_embeddings, bank.patch_embeddings).reshape(
                (-1, *grid_size)
            )
        ]

        # Window-level visual association scores for each scale
        for window_emb, ref_emb, mask in zip(
            window_embeddings,
            bank.visual_embeddings,
            self.base_model.masks,
            strict=True,
        ):
            scores = visual_association_score(window_emb, ref_emb)
            multi_scale_scores.append(harmonic_aggregation(scores, grid_size, mask))

        # Average across scales
        return torch.stack(multi_scale_scores).mean(dim=0)

    def get_image_scores(
        self, pixel_scores: torch.Tensor, selected_banks: List[str]
    ) -> torch.Tensor:
        """Compute image-level anomaly scores from pixel-level scores.

        Args:
            pixel_scores: Pixel-level anomaly scores of shape (B, H, W).
            selected_banks: List of selected bank names (unused, for interface).

        Returns:
            Image-level scores of shape (B,).
        """
        # Max over spatial dimensions
        return pixel_scores.amax(dim=(-2, -1))

    def get_gating_accuracy(
        self, selected_banks: List[str], indices: List[int]
    ) -> Tuple[float, int, int]:
        """Compute gating accuracy against oracle ground truth.

        Args:
            selected_banks: List of selected bank names from gating.
            indices: List of sample indices for ground truth lookup.

        Returns:
            Tuple of (accuracy, correct_count, total_count).
        """
        gt_banks = [self.oracle_gating.get_gt_condition(idx) for idx in indices]
        correct = sum(s == g for s, g in zip(selected_banks, gt_banks))
        total = len(indices)
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, correct, total
