# Pruning Strategy Base Class
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..config import VTRConfig


class PruningStrategy(ABC):
    """
    Abstract base class for pruning strategies.

    All strategies must implement compute_scores().

    Token selection priority (highest to lowest):
    1. keep_tokens: keep an exact number of tokens
    2. keep_ratio: keep a fraction of tokens
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
            attention: attention weights [batch, heads, seq, seq]
            image_token_range: position range of image tokens (start, end)
            config: VTR config
            **ctx: extra context (e.g. prior_attention for InfoVTR)

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
        1. keep_tokens: keep exact count
        2. keep_ratio: keep Top-K by ratio

        Args:
            scores: per-image-token scores [num_image_tokens]
            num_tokens: total number of image tokens
            config: VTR config

        Returns:
            keep_indices: indices of tokens to keep (relative to img_start) [num_keep]
        """
        device = scores.device

        # Priority 1: keep_tokens (exact count)
        if config.keep_tokens is not None:
            k = min(config.keep_tokens, num_tokens)
            if k == 0:
                keep_indices = torch.tensor([], dtype=torch.long, device=device)
            else:
                keep_indices = scores.topk(k).indices

        # Priority 2: keep_ratio
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
            config: VTR config

        Returns:
            scores: [num_image_tokens]
        """
        img_start, img_end = image_token_range

        # Select query tokens based on query_aggregation mode
        if config.query_aggregation == "last":
            attn_to_img = attention[:, :, -1, img_start:img_end]  # [1, heads, num_img]
        elif config.query_aggregation == "question":
            question_start = img_end
            question_end = attention.shape[2] - 1
            if question_end > question_start:
                attn_to_img = attention[:, :, question_start:question_end, img_start:img_end]
                attn_to_img = attn_to_img.mean(dim=2)  # [1, heads, num_img]
            else:
                # fallback to last
                attn_to_img = attention[:, :, -1, img_start:img_end]
        else:
            # default: last
            attn_to_img = attention[:, :, -1, img_start:img_end]

        # Aggregate across heads
        if config.head_aggregation == "mean":
            scores = attn_to_img.mean(dim=1).squeeze(0)  # [num_img]
        elif config.head_aggregation == "max":
            scores = attn_to_img.max(dim=1).values.squeeze(0)  # [num_img]
        else:
            scores = attn_to_img.mean(dim=1).squeeze(0)

        return scores
