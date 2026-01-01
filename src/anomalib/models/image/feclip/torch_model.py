"""PyTorch model implementation of FE-CLIP for zero-shot anomaly detection.

This module provides the core PyTorch model implementation of FE-CLIP, which uses
CLIP embeddings enhanced with frequency information through FFE and LFS adapters.

Paper: FE-CLIP: Frequency Enhanced CLIP Model for Zero-Shot Anomaly Detection

Example:
    >>> from anomalib.models.image.feclip.torch_model import FEClipModel
    >>> model = FEClipModel()  # doctest: +SKIP
    >>> model.setup_text()  # doctest: +SKIP
    >>> prediction = model(image)  # doctest: +SKIP
"""

from typing import TYPE_CHECKING

import torch
from lightning_utilities.core.imports import module_available
from torch import nn
from torch.nn import functional as F

from anomalib.data import InferenceBatch
from anomalib.models.components import BufferListMixin, DynamicBufferMixin

if TYPE_CHECKING or module_available("open_clip"):
    import open_clip
    from open_clip.tokenizer import tokenize
else:
    msg = "open_clip is required for VLM models. Install it with: pip install anomalib[vlm_clip]"
    raise ImportError(msg)

from .adapters import FFEAdapter, LFSAdapter
from .prompting import create_feclip_prompts

# Default backbone for FE-CLIP (ViT-L-14 with 336px input)
BACKBONE = "ViT-L-14-336"
PRETRAINED = "openai"
TEMPERATURE = 0.07


class FEClipModel(DynamicBufferMixin, BufferListMixin, nn.Module):
    """PyTorch module that implements the FE-CLIP model for image anomaly detection.

    The model uses CLIP embeddings enhanced with frequency-aware adapters (FFE and LFS)
    to detect anomalies in images through zero-shot learning.

    Args:
        backbone: CLIP backbone model name. Default: "ViT-L-14-336".
        pretrained: Pretrained weights to use. Default: "openai".
        n_taps: Number of tap blocks for adapter injection. Default: 4.
        lambda_fuse: Fusion weight for adapter outputs. Default: 0.1.
        P: Window size for FFE adapter. Default: 3.
        Q: Window size for LFS adapter. Default: 3.
        temperature: Temperature for softmax scaling. Default: 0.07.
        init_fc_patch_from_clip: Initialize fc_patch from visual.proj. Default: False.
            WARNING: Setting True HURTS zero-shot pixel performance significantly.

    Attributes:
        clip: CLIP model for image and text encoding.
        visual: Visual encoder of CLIP.
        ffe: ModuleList of FFE adapters for each tap block.
        lfs: ModuleList of LFS adapters for each tap block.
        fc_patch: Learnable projection from patch tokens to text space.
        fc_clip: Frozen CLIP projection for class token.
    """

    def __init__(
        self,
        backbone: str = BACKBONE,
        pretrained: str = PRETRAINED,
        n_taps: int = 4,
        tap_indices: list[int] | None = None,
        lambda_fuse: float = 0.1,
        P: int = 3,
        Q: int = 3,
        lfs_agg_mode: str = "mean",
        temperature: float = TEMPERATURE,
        use_clip_logit_scale: bool = False,
        freeze_fc_patch: bool = False,
        init_fc_patch_from_clip: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.temperature = temperature
        self.use_clip_logit_scale = use_clip_logit_scale
        self.freeze_fc_patch = freeze_fc_patch
        self.lambda_fuse = lambda_fuse
        self.n_taps = n_taps
        self.P = P
        self.Q = Q
        self.lfs_agg_mode = lfs_agg_mode

        # Initialize CLIP model
        self.clip, _, self._transform = open_clip.create_model_and_transforms(
            backbone,
            pretrained=pretrained,
        )
        self.visual = self.clip.visual

        # Freeze CLIP parameters
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        # Determine tap blocks
        resblocks = self.visual.transformer.resblocks
        n_blocks = len(resblocks)
        if tap_indices is not None:
            # Use custom tap indices
            self.tap_blocks = [i for i in tap_indices if 0 <= i < n_blocks]
        elif n_taps >= n_blocks:
            self.tap_blocks = list(range(n_blocks))
        else:
            # Default: evenly distributed across ViT blocks
            self.tap_blocks = torch.linspace(0, n_blocks - 1, steps=n_taps).round().long().tolist()

        # Get model dimensions
        width = self.visual.transformer.width  # token dimension

        # Create FFE and LFS adapters for each tap block
        self.ffe = nn.ModuleList([FFEAdapter(width, P=P) for _ in self.tap_blocks])
        self.lfs = nn.ModuleList([LFSAdapter(width, Q=Q, agg_mode=lfs_agg_mode) for _ in self.tap_blocks])

        # Text embedding buffer
        self.register_buffer("_text_emb", torch.empty(0))

        # Learnable patch projection to text space
        text_dim = self._get_text_dim()
        self.fc_patch = nn.Linear(width, text_dim)

        # Optional: Initialize fc_patch from frozen CLIP projection (visual.proj)
        # WARNING: This HURTS zero-shot pixel performance significantly (0.74 → 0.18)
        # Only enable for specific finetuning experiments where it may help.
        # Default is False (random init) which works better for zero-shot.
        if init_fc_patch_from_clip:
            with torch.no_grad():
                W = self.visual.proj  # (width, text_dim) in open_clip
                # fc_patch.weight expects (text_dim, width) for nn.Linear
                if W.shape == (width, text_dim):
                    self.fc_patch.weight.copy_(W.T)
                elif W.shape == (text_dim, width):
                    self.fc_patch.weight.copy_(W)
                self.fc_patch.bias.data.zero_()

        # Optional: Freeze fc_patch (only train adapters)
        if freeze_fc_patch:
            for p in self.fc_patch.parameters():
                p.requires_grad = False

        # fc_clip is FROZEN (original CLIP projection) per paper
        # "fc_clip denotes the frozen project layer"
        # visual.proj is already frozen (no grad) since we froze all CLIP params above

        # Grid size for token reshaping
        self._grid_size = self.visual.grid_size

    def train(self, mode: bool = True) -> "FEClipModel":
        """Override train to keep CLIP always in eval mode.

        This is critical because:
        - CLIP has dropout layers that activate in train mode
        - Even with frozen weights, dropout causes unstable token distributions
        - Adapters learn from "noisy teacher" leading to degraded performance

        Args:
            mode: Whether to set training mode (True) or eval mode (False).

        Returns:
            Self.
        """
        super().train(mode)
        # Always keep CLIP components in eval mode to disable dropout
        self.clip.eval()
        self.visual.eval()
        return self

    def _get_text_dim(self) -> int:
        """Get text embedding dimension from CLIP model."""
        if hasattr(self.clip, "text_projection") and self.clip.text_projection is not None:
            return self.clip.text_projection.shape[1]
        return self.visual.proj.shape[1]

    @property
    def grid_size(self) -> tuple[int, int]:
        """Get the grid size of the visual encoder."""
        return self._grid_size

    def setup_text(self) -> None:
        """Setup text embeddings from prompts.

        This method must be called before forward pass to initialize
        the text embeddings used for anomaly score computation.
        """
        device = next(self.parameters()).device
        normal_prompt, abnormal_prompt = create_feclip_prompts()
        prompts = [normal_prompt, abnormal_prompt]
        tok = tokenize(prompts).to(device)
        with torch.no_grad():
            t = self.clip.encode_text(tok)
            t = F.normalize(t, dim=-1)
        self._text_emb = t  # (2, text_dim)

    @property
    def text_emb(self) -> torch.Tensor:
        """Get text embeddings.

        Returns:
            Text embeddings of shape (2, text_dim).

        Raises:
            RuntimeError: If setup_text() has not been called.
        """
        if self._text_emb.numel() == 0:
            msg = "Text embeddings not initialized. Call setup_text() before forward."
            raise RuntimeError(msg)
        return self._text_emb

    def prob_abnormal(self, z: torch.Tensor) -> torch.Tensor:
        """Compute probability of being abnormal.

        Args:
            z: Feature tensor of shape (..., text_dim).

        Returns:
            Probability of abnormal class of shape (...).
        """
        z = F.normalize(z, dim=-1)
        if self.use_clip_logit_scale:
            # Use CLIP's learned logit_scale (typically ~4.6, equivalent to τ≈0.01)
            logit_scale = self.clip.logit_scale.exp()
            logits = logit_scale * (z @ self.text_emb.t())
        else:
            # Use fixed temperature (default 0.07)
            logits = (z @ self.text_emb.t()) / self.temperature
        return logits.softmax(dim=-1)[..., 1]

    def forward_tokens(
        self,
        images: torch.Tensor,
        valid_mask_tokens: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass through ViT blocks with adapter injection.

        Args:
            images: Input images of shape (B, C, H, W).
            valid_mask_tokens: Optional mask for valid token regions of shape (B, Ht, Wt).
                Used to handle letterbox padding regions.

        Returns:
            Tuple of (score_list, map_list) where:
                - score_list: List of anomaly scores from each tap block, each (B,).
                - map_list: List of anomaly maps for each tap block, each (B, Ht, Wt).
        """
        # Stem/tokenize (patch embedding)
        x = self.visual.conv1(images)  # (B, C', H', W')
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, HW, D)

        # Add class token
        cls = self.visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)  # (B, 1+HW, D)

        # Add positional embedding and layer norm
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        # Grid size for reshaping
        Ht, Wt = self.grid_size
        B = x.shape[0]

        score_list = []
        map_list = []
        tap_i = 0

        # IMPORTANT: CLIP transformer expects (L, N, D) format, not (N, L, D)
        x = x.permute(1, 0, 2)  # (B, L, D) -> (L, B, D)

        # Process through transformer blocks
        for bi, blk in enumerate(self.visual.transformer.resblocks):
            x = blk(x)

            if bi in self.tap_blocks:
                # Permute back to (B, L, D) for adapter operations
                x_bld = x.permute(1, 0, 2)  # (L, B, D) -> (B, L, D)

                # Extract patch tokens and reshape to spatial grid
                patch = x_bld[:, 1:, :].reshape(B, Ht, Wt, -1)

                # Apply FFE and LFS adapters
                ffe_out = self.ffe[tap_i](patch)
                lfs_out = self.lfs[tap_i](patch, valid_mask=valid_mask_tokens)

                # Fuse adapter outputs with original features
                patch_hat = self.lambda_fuse * (ffe_out + lfs_out) + (1 - self.lambda_fuse) * patch

                # Update token sequence with fused features
                x_bld = torch.cat([x_bld[:, :1, :], patch_hat.reshape(B, -1, x_bld.shape[-1])], dim=1)

                # Permute back to (L, B, D) for next transformer block
                x = x_bld.permute(1, 0, 2)

                # Compute patch-level anomaly map at this tap block
                # IMPORTANT: Apply ln_post to patch tokens for consistent normalization
                # (cls token also goes through ln_post before projection)
                patch_flat = patch_hat.reshape(B, -1, patch_hat.shape[-1])  # (B, Ht*Wt, D)
                patch_norm = self.visual.ln_post(patch_flat)  # Apply same normalization as cls
                patch_norm = patch_norm.reshape(B, Ht, Wt, -1)  # (B, Ht, Wt, D)

                patch_txt = self.fc_patch(patch_norm)  # (B, Ht, Wt, text_dim)
                amap = self.prob_abnormal(patch_txt)  # (B, Ht, Wt)
                map_list.append(amap)

                # Compute image-level score at this tap block (Equation 3 in paper)
                # fc_clip is FROZEN: use visual.proj directly
                cls_tok = x_bld[:, 0, :]  # (B, D)
                cls_tok = self.visual.ln_post(cls_tok)
                cls_txt = cls_tok @ self.visual.proj  # Frozen projection (D, text_dim)
                score_n = self.prob_abnormal(cls_txt)  # (B,)
                score_list.append(score_n)

                tap_i += 1

        return score_list, map_list

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        valid_mask_tokens: torch.Tensor | None = None,
    ) -> InferenceBatch:
        """Forward pass for inference.

        Args:
            images: Input images of shape (B, C, H, W).
            valid_mask_tokens: Optional mask for valid token regions of shape (B, Ht, Wt).

        Returns:
            InferenceBatch containing pred_score and anomaly_map.
        """
        scores, maps = self.forward_tokens(images, valid_mask_tokens)

        # Average scores across tap blocks (Equation 3 in paper: Sa = 1/N * Σ Sa,n)
        score = torch.stack(scores, dim=0).mean(dim=0)  # (B,)

        # Average anomaly maps across tap blocks (Equation 3 in paper)
        amap = torch.stack(maps, dim=0).mean(dim=0)  # (B, Ht, Wt)

        # Upsample anomaly map to input resolution
        amap = F.interpolate(
            amap.unsqueeze(1),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return InferenceBatch(pred_score=score, anomaly_map=amap)

    def forward_for_training(
        self,
        images: torch.Tensor,
        valid_mask_tokens: torch.Tensor | None = None,
    ) -> InferenceBatch:
        """Forward pass for training (with gradients).

        Args:
            images: Input images of shape (B, C, H, W).
            valid_mask_tokens: Optional mask for valid token regions of shape (B, Ht, Wt).

        Returns:
            InferenceBatch containing pred_score and anomaly_map.
        """
        scores, maps = self.forward_tokens(images, valid_mask_tokens)

        # Average scores across tap blocks (Equation 3 in paper: Sa = 1/N * Σ Sa,n)
        score = torch.stack(scores, dim=0).mean(dim=0)  # (B,)

        # Average anomaly maps across tap blocks
        amap = torch.stack(maps, dim=0).mean(dim=0)  # (B, Ht, Wt)

        # Upsample anomaly map to input resolution
        amap = F.interpolate(
            amap.unsqueeze(1),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return InferenceBatch(pred_score=score, anomaly_map=amap)
