"""Multi-condition gating for Condition-Aware WinCLIP.

This module provides multiple gating strategies for condition-aware bank selection:
1. MultiConditionGating: CLIP embedding-based Top-K similarity
2. P90IntensityGating: Intensity-based p90 percentile with midpoint threshold
3. OracleGating: Ground truth based (for evaluation)

The P90IntensityGating is recommended for HDMAP dataset as it achieves 96.7% accuracy
compared to 88.8% for CLIP-based gating.
"""

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def cosine_similarity_simple(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two tensors.

    Args:
        input1: Query tensor of shape (D,) or (1, D)
        input2: Reference tensor of shape (N, D)

    Returns:
        Similarity scores of shape (N,)
    """
    if input1.dim() == 1:
        input1 = input1.unsqueeze(0)  # (1, D)

    input1_norm = torch.nn.functional.normalize(input1, p=2, dim=-1)  # (1, D)
    input2_norm = torch.nn.functional.normalize(input2, p=2, dim=-1)  # (N, D)

    # (1, D) @ (D, N) -> (1, N) -> (N,)
    similarity = torch.mm(input1_norm, input2_norm.T).squeeze(0)
    return similarity


class MultiConditionGating:
    """N-way condition-aware bank selection using Top-K global similarity.

    This class selects the most appropriate reference bank for a test image
    based on global embedding similarity. While currently used for binary
    (cold/warm) selection, it supports arbitrary N banks.

    Attributes:
        bank_embeddings: Dictionary mapping bank names to global embeddings.
        k: Number of top references to average for similarity score.

    Example:
        >>> bank_emb = {"cold": torch.randn(2, 640), "warm": torch.randn(2, 640)}
        >>> gating = MultiConditionGating(bank_emb, k=1)
        >>> test_emb = torch.randn(640)
        >>> selected, scores = gating.select_bank(test_emb)
        >>> print(f"Selected: {selected}, Scores: {scores}")
    """

    def __init__(self, bank_embeddings: Dict[str, torch.Tensor], k: int = 1, verbose: bool = False):
        """Initialize the gating mechanism.

        Args:
            bank_embeddings: Dictionary mapping bank names to global embeddings.
                Each value should be a tensor of shape (N_refs, D) where N_refs
                is the number of reference images and D is the embedding dimension.
            k: Number of top references to average when computing bank similarity.
                For small reference sets (1-2), use k=1.
            verbose: If True, log detailed similarity scores for each gating decision.
        """
        self.bank_embeddings = bank_embeddings
        self.k = k
        self.bank_names = list(bank_embeddings.keys())
        self.verbose = verbose

        # Statistics tracking
        self.call_count = 0
        self.score_history = []  # Store recent scores for analysis

        logger.info(f"MultiConditionGating initialized with k={k}, verbose={verbose}")
        for name, emb in bank_embeddings.items():
            logger.info(f"  Bank '{name}': {emb.shape[0]} refs, dim={emb.shape[1]}")

    def select_bank(self, test_embedding: torch.Tensor, file_index: int = None) -> Tuple[str, Dict[str, float]]:
        """Select the most appropriate bank for a test image.

        Computes Top-K average similarity between the test embedding and each
        bank's reference embeddings, then selects the bank with highest score.

        Args:
            test_embedding: Global embedding of the test image, shape (D,).
            file_index: Optional file index for logging purposes.

        Returns:
            Tuple of (selected_bank_name, {bank_name: similarity_score}).
            The scores dictionary contains the Top-K average similarity for each bank.
        """
        scores = {}
        raw_sims = {}  # Store raw similarities for verbose logging

        for bank_name, bank_emb in self.bank_embeddings.items():
            # Compute similarity with all references in this bank
            sim = cosine_similarity_simple(test_embedding, bank_emb)  # (N_refs,)
            raw_sims[bank_name] = sim.cpu().numpy().tolist()

            # Take Top-K average (handles case where k > N_refs)
            top_k = min(self.k, len(sim))
            scores[bank_name] = sim.topk(top_k)[0].mean().item()

        # Select bank with highest similarity
        selected = max(scores, key=scores.get)

        # Calculate margin
        score_values = list(scores.values())
        margin = abs(score_values[0] - score_values[1]) if len(score_values) >= 2 else 0

        # Track statistics
        self.call_count += 1
        self.score_history.append({
            'file_index': file_index,
            'scores': scores.copy(),
            'selected': selected,
            'margin': margin
        })

        # Verbose logging for first 10 samples, then every 100th
        if self.verbose and (self.call_count <= 10 or self.call_count % 100 == 0):
            gt_condition = "cold" if file_index is not None and file_index < 500 else "warm"
            correct = "✓" if selected == gt_condition else "✗"
            logger.info(
                f"  [Gating #{self.call_count}] file={file_index} | "
                f"cold={scores.get('cold', 0):.4f}, warm={scores.get('warm', 0):.4f} | "
                f"margin={margin:.4f} | selected={selected} (GT={gt_condition}) {correct}"
            )

        return selected, scores

    def get_score_statistics(self) -> Dict:
        """Get statistics about gating scores for analysis."""
        if not self.score_history:
            return {}

        cold_scores = [h['scores'].get('cold', 0) for h in self.score_history]
        warm_scores = [h['scores'].get('warm', 0) for h in self.score_history]
        margins = [h['margin'] for h in self.score_history]

        import numpy as np
        return {
            'cold_score_mean': np.mean(cold_scores),
            'cold_score_std': np.std(cold_scores),
            'warm_score_mean': np.mean(warm_scores),
            'warm_score_std': np.std(warm_scores),
            'margin_mean': np.mean(margins),
            'margin_std': np.std(margins),
            'total_samples': len(self.score_history)
        }

    def select_bank_batch(
        self, test_embeddings: torch.Tensor
    ) -> Tuple[list, list]:
        """Select banks for a batch of test images.

        Args:
            test_embeddings: Batch of global embeddings, shape (B, D).

        Returns:
            Tuple of (list of selected bank names, list of score dicts).
        """
        selected_banks = []
        all_scores = []

        for emb in test_embeddings:
            selected, scores = self.select_bank(emb)
            selected_banks.append(selected)
            all_scores.append(scores)

        return selected_banks, all_scores


class OracleGating:
    """Oracle gating that uses ground truth condition labels.

    This class is for evaluation purposes - it always selects the correct
    bank based on provided ground truth labels.

    Example:
        >>> gating = OracleGating()
        >>> # For index 100 (cold), returns "cold"
        >>> selected = gating.select_bank_by_index(100)
    """

    def __init__(self, cold_indices: range = range(0, 500), warm_indices: range = range(500, 1000)):
        """Initialize oracle gating with index ranges.

        Args:
            cold_indices: Range of indices considered "cold" (default 0-499).
            warm_indices: Range of indices considered "warm" (default 500-999).
        """
        self.cold_indices = cold_indices
        self.warm_indices = warm_indices

    def select_bank_by_index(self, idx: int) -> str:
        """Select bank based on sample index.

        Args:
            idx: Sample index in the dataset.

        Returns:
            "cold" if index is in cold range, "warm" otherwise.
        """
        return "cold" if idx in self.cold_indices else "warm"

    def get_gt_condition(self, idx: int) -> str:
        """Get ground truth condition for an index (alias for select_bank_by_index)."""
        return self.select_bank_by_index(idx)


class P90IntensityGating:
    """Intensity-based gating using p90 percentile with midpoint threshold.

    This gating method uses the 90th percentile of image pixel intensities
    to determine whether an image is from cold or warm condition.

    The threshold is computed as the midpoint between:
    - max(p90 of good/cold images)
    - min(p90 of good/warm images)

    This method achieves ~96.7% gating accuracy on HDMAP dataset,
    significantly outperforming CLIP-based gating (~88.8%).

    Attributes:
        threshold: The midpoint threshold for cold/warm decision.
        cold_p90_max: Maximum p90 value from cold reference images.
        warm_p90_min: Minimum p90 value from warm reference images.

    Example:
        >>> # Compute threshold from reference images
        >>> cold_images = [...]  # List of cold reference image tensors
        >>> warm_images = [...]  # List of warm reference image tensors
        >>> gating = P90IntensityGating.from_reference_images(cold_images, warm_images)
        >>> # Or set threshold directly
        >>> gating = P90IntensityGating(threshold=0.3089)
        >>> # Select condition for test image
        >>> condition = gating.select_bank(test_image_tensor)
    """

    # Pre-computed thresholds for each domain (midpoint strategy)
    DOMAIN_THRESHOLDS = {
        'domain_A': 0.2985,
        'domain_B': 0.3128,
        'domain_C': 0.3089,
        'domain_D': 0.2919,
    }

    def __init__(
        self,
        threshold: float = None,
        domain: str = None,
        verbose: bool = False,
    ):
        """Initialize P90 intensity gating.

        Args:
            threshold: The p90 threshold for cold/warm decision.
                If image p90 <= threshold -> cold, else -> warm.
            domain: Domain name to use pre-computed threshold.
                One of 'domain_A', 'domain_B', 'domain_C', 'domain_D'.
            verbose: If True, log detailed gating decisions.

        Note:
            Either threshold or domain must be provided.
        """
        if threshold is not None:
            self.threshold = threshold
        elif domain is not None:
            if domain not in self.DOMAIN_THRESHOLDS:
                raise ValueError(f"Unknown domain: {domain}. Available: {list(self.DOMAIN_THRESHOLDS.keys())}")
            self.threshold = self.DOMAIN_THRESHOLDS[domain]
        else:
            raise ValueError("Either 'threshold' or 'domain' must be provided")

        self.verbose = verbose
        self.cold_p90_max = None
        self.warm_p90_min = None

        # Statistics tracking
        self.call_count = 0
        self.p90_history = []

        logger.info(f"P90IntensityGating initialized with threshold={self.threshold:.4f}")

    @classmethod
    def from_reference_images(
        cls,
        cold_images: List[Union[torch.Tensor, np.ndarray]],
        warm_images: List[Union[torch.Tensor, np.ndarray]],
        verbose: bool = False,
    ) -> "P90IntensityGating":
        """Create gating from reference images using midpoint threshold.

        Computes threshold as: (max(cold_p90) + min(warm_p90)) / 2

        Args:
            cold_images: List of cold reference images (tensors or numpy arrays).
            warm_images: List of warm reference images (tensors or numpy arrays).
            verbose: If True, log detailed gating decisions.

        Returns:
            P90IntensityGating instance with computed threshold.
        """
        cold_p90_values = [cls._compute_p90(img) for img in cold_images]
        warm_p90_values = [cls._compute_p90(img) for img in warm_images]

        cold_p90_max = max(cold_p90_values)
        warm_p90_min = min(warm_p90_values)
        threshold = (cold_p90_max + warm_p90_min) / 2

        logger.info(f"P90 threshold computed from {len(cold_images)} cold + {len(warm_images)} warm images")
        logger.info(f"  cold_p90_max={cold_p90_max:.4f}, warm_p90_min={warm_p90_min:.4f}")
        logger.info(f"  midpoint threshold={threshold:.4f}")

        instance = cls(threshold=threshold, verbose=verbose)
        instance.cold_p90_max = cold_p90_max
        instance.warm_p90_min = warm_p90_min
        return instance

    @staticmethod
    def _compute_p90(image: Union[torch.Tensor, np.ndarray]) -> float:
        """Compute 90th percentile of image pixels.

        Args:
            image: Image tensor (C, H, W) or numpy array.

        Returns:
            90th percentile value.
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        return float(np.percentile(image, 90))

    def select_bank(
        self,
        image: Union[torch.Tensor, np.ndarray],
        file_index: int = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Select condition (cold/warm) based on image p90.

        Args:
            image: Test image tensor (C, H, W) or numpy array.
            file_index: Optional file index for logging.

        Returns:
            Tuple of (selected_condition, {"p90": value, "threshold": threshold}).
        """
        p90 = self._compute_p90(image)
        selected = "cold" if p90 <= self.threshold else "warm"

        # Track statistics
        self.call_count += 1
        self.p90_history.append({
            'file_index': file_index,
            'p90': p90,
            'selected': selected,
            'margin': abs(p90 - self.threshold),
        })

        # Verbose logging
        if self.verbose and (self.call_count <= 10 or self.call_count % 100 == 0):
            gt_condition = "cold" if file_index is not None and file_index < 500 else "warm"
            correct = "OK" if selected == gt_condition else "WRONG"
            logger.info(
                f"  [P90 Gating #{self.call_count}] file={file_index} | "
                f"p90={p90:.4f}, threshold={self.threshold:.4f} | "
                f"selected={selected} (GT={gt_condition}) {correct}"
            )

        scores = {
            'p90': p90,
            'threshold': self.threshold,
            'cold': 1.0 if selected == 'cold' else 0.0,
            'warm': 1.0 if selected == 'warm' else 0.0,
        }

        return selected, scores

    def select_bank_batch(
        self,
        images: List[Union[torch.Tensor, np.ndarray]],
    ) -> Tuple[List[str], List[Dict[str, float]]]:
        """Select conditions for a batch of images.

        Args:
            images: List of test images.

        Returns:
            Tuple of (list of selected conditions, list of score dicts).
        """
        selected_banks = []
        all_scores = []

        for img in images:
            selected, scores = self.select_bank(img)
            selected_banks.append(selected)
            all_scores.append(scores)

        return selected_banks, all_scores

    def get_statistics(self) -> Dict:
        """Get statistics about gating decisions."""
        if not self.p90_history:
            return {}

        p90_values = [h['p90'] for h in self.p90_history]
        margins = [h['margin'] for h in self.p90_history]
        cold_count = sum(1 for h in self.p90_history if h['selected'] == 'cold')

        return {
            'p90_mean': np.mean(p90_values),
            'p90_std': np.std(p90_values),
            'p90_min': np.min(p90_values),
            'p90_max': np.max(p90_values),
            'margin_mean': np.mean(margins),
            'threshold': self.threshold,
            'cold_ratio': cold_count / len(self.p90_history),
            'total_samples': len(self.p90_history),
        }


class RandomGating:
    """Random gating for baseline comparison.

    Randomly selects cold or warm bank with 50% probability.
    Used to establish a baseline for gating performance.

    Example:
        >>> gating = RandomGating()
        >>> selected, scores = gating.select_bank(image)
    """

    def __init__(self, seed: int = None):
        """Initialize random gating.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)
        self.call_count = 0
        self.history = []

        logger.info(f"RandomGating initialized (seed={seed})")

    def select_bank(
        self,
        image_or_embedding,
        file_index: int = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Randomly select cold or warm bank.

        Args:
            image_or_embedding: Unused (any input accepted for interface compatibility).
            file_index: Optional file index for logging.

        Returns:
            Tuple of (selected_condition, {"random": True}).
        """
        selected = "cold" if self.rng.random() < 0.5 else "warm"

        self.call_count += 1
        self.history.append({
            'file_index': file_index,
            'selected': selected,
        })

        scores = {
            'cold': 0.5,
            'warm': 0.5,
            'random': True,
        }

        return selected, scores


class InverseGating:
    """Inverse gating for worst-case baseline.

    Always selects the wrong bank (cold→warm, warm→cold).
    Used to establish the worst-case performance bound.

    Example:
        >>> gating = InverseGating()
        >>> selected, scores = gating.select_bank(image, file_index=123)
    """

    def __init__(self, cold_indices: range = range(0, 500), warm_indices: range = range(500, 1000)):
        """Initialize inverse gating.

        Args:
            cold_indices: Range of indices considered "cold" (default 0-499).
            warm_indices: Range of indices considered "warm" (default 500-999).
        """
        self.cold_indices = cold_indices
        self.warm_indices = warm_indices
        self.call_count = 0
        self.history = []

        logger.info("InverseGating initialized (always selects wrong bank)")

    def select_bank(
        self,
        image_or_embedding,
        file_index: int = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Select the wrong bank based on file index.

        Args:
            image_or_embedding: Unused (any input accepted for interface compatibility).
            file_index: File index to determine GT condition (then select opposite).

        Returns:
            Tuple of (selected_condition, {"inverse": True}).
        """
        if file_index is None:
            # If no index provided, randomly select (shouldn't happen in practice)
            selected = "cold"
        else:
            # Determine GT condition, then select opposite
            gt_condition = "cold" if (file_index % 1000) in self.cold_indices else "warm"
            selected = "warm" if gt_condition == "cold" else "cold"

        self.call_count += 1
        self.history.append({
            'file_index': file_index,
            'selected': selected,
        })

        scores = {
            'cold': 1.0 if selected == 'cold' else 0.0,
            'warm': 1.0 if selected == 'warm' else 0.0,
            'inverse': True,
        }

        return selected, scores
