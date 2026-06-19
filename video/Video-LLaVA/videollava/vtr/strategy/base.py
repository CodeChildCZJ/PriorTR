# Pruning Strategy Base Class
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..config import VTRConfig, InfoVTRConfig


class PruningStrategy(ABC):
    """
    Abstract base class for pruning strategies.

    All strategies must implement the compute_scores method.

    Token selection priority (highest to lowest):
    1. keep_tokens: keep an exact number of tokens
    2. score_threshold: keep tokens with score above threshold (InfoVTRConfig only)
    3. keep_ratio: keep tokens by ratio
    """

    @abstractmethod
    def compute_scores(
        self,
        attention: torch.Tensor,
        image_token_range: Tuple[int, int],
        config: VTRConfig,
        **ctx,
    ) -> torch.Tensor:
        """
        Compute importance scores for each image token.

        Args:
            attention: attention weights at current layer [batch, heads, seq, seq]
            image_token_range: position range of image tokens (start, end)
            config: VTR configuration
            **ctx: extra context (e.g., InfoVTR needs prior_attention)

        Returns:
            scores: per-image-token scores [num_image_tokens]
        """
        pass

    def select_tokens(
        self,
        scores: torch.Tensor,
        num_tokens: int,
        config: VTRConfig,
    ) -> torch.Tensor:
        """
        Select token indices to keep based on scores.

        Selection priority (highest to lowest):
        1. keep_tokens: keep an exact number of tokens
        2. score_threshold: keep tokens with score > threshold (InfoVTRConfig only)
        3. keep_ratio: keep Top-K by ratio

        Args:
            scores: per-image-token scores [num_image_tokens]
            num_tokens: total number of image tokens
            config: VTR configuration

        Returns:
            keep_indices: indices of tokens to keep (relative to image_start) [num_keep]
        """
        device = scores.device

        # Priority 1: keep_tokens (exact count)
        if config.keep_tokens is not None:
            k = min(config.keep_tokens, num_tokens)
            if k == 0:
                keep_indices = torch.tensor([], dtype=torch.long, device=device)
            else:
                keep_indices = scores.topk(k).indices

        # Priority 2: score_threshold (threshold mode, InfoVTRConfig only)
        elif hasattr(config, "score_threshold") and config.score_threshold is not None:
            keep_indices = torch.where(scores > config.score_threshold)[0]
            # If no token meets the threshold, keep at least the highest-scoring one
            if len(keep_indices) == 0:
                keep_indices = scores.topk(1).indices

        # Priority 3: keep_ratio (ratio mode)
        else:
            k = max(0, int(num_tokens * config.keep_ratio))
            if k == 0:
                keep_indices = torch.tensor([], dtype=torch.long, device=device)
            else:
                keep_indices = scores.topk(k).indices

        # Sort to preserve original order
        keep_indices = keep_indices.sort().values
        return keep_indices

    def _aggregate_attention(
        self,
        attention: torch.Tensor,
        image_token_range: Tuple[int, int],
        config: VTRConfig,
    ) -> torch.Tensor:
        """
        Aggregate attention into per-image-token scores.

        Args:
            attention: [batch, heads, seq, seq]
            image_token_range: (start, end)
            config: VTR configuration

        Returns:
            scores: [num_image_tokens]
        """
        img_start, img_end = image_token_range

        # Select query tokens based on query_aggregation
        if config.query_aggregation == "last":
            # Last token's attention to image tokens
            attn_to_img = attention[:, :, -1, img_start:img_end]  # [1, heads, num_img]
        elif config.query_aggregation == "question":
            # Average attention from question tokens (after image, before last) to image
            question_start = img_end
            question_end = attention.shape[2] - 1
            if question_end > question_start:
                attn_to_img = attention[:, :, question_start:question_end, img_start:img_end]
                attn_to_img = attn_to_img.mean(dim=2)  # [1, heads, num_img]
            else:
                # fallback to last
                attn_to_img = attention[:, :, -1, img_start:img_end]
        else:
            # Default: use last
            attn_to_img = attention[:, :, -1, img_start:img_end]

        # Aggregate heads based on head_aggregation
        if config.head_aggregation == "mean":
            scores = attn_to_img.mean(dim=1).squeeze(0)  # [num_img]
        elif config.head_aggregation == "max":
            scores = attn_to_img.max(dim=1).values.squeeze(0)  # [num_img]
        else:
            scores = attn_to_img.mean(dim=1).squeeze(0)

        return scores
