"""Text prompting utilities for FE-CLIP model.

This module provides text prompt generation for zero-shot anomaly detection.

Paper: FE-CLIP: Frequency Enhanced CLIP Model for Zero-Shot Anomaly Detection
"""


def create_feclip_prompts() -> tuple[str, str]:
    """Create normal and abnormal text prompts for FE-CLIP.

    Uses the exact prompts from the original FE-CLIP paper.

    Returns:
        Tuple of (normal_prompt, abnormal_prompt).
    """
    # Original FE-CLIP paper prompts
    normal_prompt = "A photo of a normal object"
    abnormal_prompt = "A photo of a damaged object"
    return normal_prompt, abnormal_prompt
