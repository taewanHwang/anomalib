"""
Dinomaly model with Adaptive Bottleneck Dropout using APE.

This variant of Dinomaly uses Angular Power Entropy (APE)-based adaptive dropout
to prevent overfitting to highly regular normal patterns during long training.

APE measures directional concentration of power in the frequency domain:
- Low APE (strong directional pattern) → Higher overfit risk → Higher dropout
- High APE (isotropic/complex pattern) → Lower overfit risk → Lower dropout
"""

import torch
import torch.nn as nn

from .torch_model import DinomalyModel
from .adaptive_dropout import (
    AdaptiveMLP,
    compute_angular_power_entropy_batch,
    AdaptiveDropoutStats,
)
from anomalib.data import InferenceBatch


class DinomalyModelAdaptive(DinomalyModel):
    """
    Dinomaly model with APE-based adaptive bottleneck dropout.

    Inherits from DinomalyModel and replaces the bottleneck MLP with AdaptiveMLP.
    The dropout probability is adjusted per-sample based on Angular Power Entropy:
    - APE < normal_ape (more directional) → Higher dropout
    - APE > normal_ape (more isotropic) → Lower dropout

    Args:
        encoder_name (str): Name of the DINOv2 encoder to use.
        base_dropout (float): Base dropout probability (at normal_ape).
        min_dropout (float): Minimum dropout probability.
        max_dropout (float): Maximum dropout probability.
        dropout_sensitivity (float): Sensitivity of APE-to-dropout mapping.
        normal_ape (float): Reference APE from normal samples.
            Domain-specific defaults from EDA:
            - domain_A: 0.777
            - domain_B: 0.713
            - domain_C: 0.866
            - domain_D: 0.816
        use_adaptive_dropout (bool): If False, uses fixed base_dropout.
        **kwargs: Additional arguments passed to DinomalyModel.
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        base_dropout: float = 0.3,
        min_dropout: float = 0.1,
        max_dropout: float = 0.6,
        dropout_sensitivity: float = 4.0,
        normal_ape: float = 0.78,
        use_adaptive_dropout: bool = True,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
    ) -> None:
        # Initialize parent class fully
        super().__init__(
            encoder_name=encoder_name,
            bottleneck_dropout=base_dropout,  # Use base_dropout as initial value
            decoder_depth=decoder_depth,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token,
        )

        # Store adaptive dropout config
        self.use_adaptive_dropout = use_adaptive_dropout
        self.dropout_stats = AdaptiveDropoutStats()

        # Get embed_dim from the existing bottleneck
        embed_dim = self.bottleneck[0].fc1.in_features

        # Replace bottleneck with AdaptiveMLP
        adaptive_mlp = AdaptiveMLP(
            in_features=embed_dim,
            hidden_features=embed_dim * 4,
            out_features=embed_dim,
            act_layer=nn.GELU,
            base_dropout=base_dropout,
            min_dropout=min_dropout,
            max_dropout=max_dropout,
            sensitivity=dropout_sensitivity,
            normal_ape=normal_ape,
            bias=False,
            apply_input_dropout=True,
            use_adaptive=use_adaptive_dropout,
        )
        self.bottleneck = nn.ModuleList([adaptive_mlp])
        self.adaptive_mlp = adaptive_mlp

    def forward(
        self,
        batch: torch.Tensor,
        global_step: int | None = None,
    ) -> torch.Tensor | InferenceBatch:
        """
        Forward pass with adaptive dropout.

        First computes APE from input images and sets adaptive dropout
        probabilities, then proceeds with normal forward pass.
        """
        # Compute and set adaptive dropout probabilities
        if self.training and self.use_adaptive_dropout:
            dropout_probs = self.adaptive_mlp.set_dropout_from_images(batch)

            # Log statistics for monitoring
            ape = compute_angular_power_entropy_batch(batch)
            self.dropout_stats.log(ape, dropout_probs)

        # Call parent forward
        return super().forward(batch, global_step)

    def get_dropout_stats(self) -> dict:
        """Get adaptive dropout statistics for monitoring."""
        return self.dropout_stats.get_summary()

    def reset_dropout_stats(self) -> None:
        """Reset dropout statistics."""
        self.dropout_stats.reset()
