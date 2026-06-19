"""SparseVLM strategy implementation.

Text-guided visual token sparsification from SparseVLMs v1.0.
Identifies important text tokens via visual-text similarity, then scores
visual tokens based on attention from important text tokens.

Reference: "Visual Token Sparsification for Efficient VLM Inference" (ICML 2025)
"""

import logging
from typing import Any, Dict, Tuple

import torch

from ..config import VTRConfig
from ..model.token_merge import cluster_and_merge
from .base import VTRStrategy

logger = logging.getLogger(__name__)


class SparseVLMStrategy(VTRStrategy):
    """SparseVLM v1.0 text-guided pruning strategy.

    Pipeline:
    1. prepare(): Identify important text tokens (above-average similarity to visual)
    2. compute_scores(): Score visual tokens by attention from important text tokens
    3. post_prune(): Optional token merge via density peak clustering
    """

    def prepare(
        self,
        hidden_states: torch.Tensor,
        config: VTRConfig,
        context: Dict[str, Any],
    ) -> None:
        """Identify text tokens that are important to visual content.

        Computes similarity between visual and text token embeddings.
        Text tokens with above-average similarity are marked as "important"
        and stored in context for use by compute_scores().
        """
        image_token_range = context.get("image_token_range")
        if image_token_range is None:
            return

        img_start, img_end = image_token_range

        v_t = hidden_states[:, img_start:img_end, :]  # [B, N_vis, D]
        t_t = hidden_states[:, img_end:, :]             # [B, N_text, D]

        if t_t.shape[1] == 0:
            logger.warning("No text tokens found for SparseVLM prepare()")
            return

        # Compute visual-text similarity
        # [B, N_vis, N_text] -> softmax along text -> mean along visual -> [B, N_text]
        sim = (v_t @ t_t.transpose(1, 2)).softmax(dim=2).mean(dim=1)

        # Select text tokens with above-average importance
        important_idx = torch.where(sim > sim.mean())

        context["important_text_token_idx"] = important_idx

        if config.debug:
            logger.debug(
                f"SparseVLM prepare: {len(important_idx[1])}/{t_t.shape[1]} "
                f"important text tokens identified"
            )

    def compute_scores(
        self,
        attention: torch.Tensor,
        image_token_range: Tuple[int, int],
        config: VTRConfig,
        layer_idx: int = 0,
        **context: Any,
    ) -> torch.Tensor:
        """Score visual tokens using attention from important text tokens.

        For each visual token, the importance score is the mean attention
        weight from all important text tokens to that visual token,
        averaged across attention heads.
        """
        img_start, img_end = image_token_range
        important_idx = context.get("important_text_token_idx")

        if important_idx is None or len(important_idx[1]) == 0:
            logger.warning(
                "No important text tokens available. "
                "Falling back to uniform scores."
            )
            num_img = img_end - img_start
            return torch.ones(num_img, device=attention.device, dtype=attention.dtype)

        # Average across attention heads
        attn = attention.mean(dim=1)  # [B, seq, seq]

        # Convert relative text indices to absolute positions
        abs_text_idx = important_idx[1] + img_end

        # Clamp to valid range
        seq_len = attn.shape[-1]
        abs_text_idx = abs_text_idx[abs_text_idx < seq_len]

        if len(abs_text_idx) == 0:
            num_img = img_end - img_start
            return torch.ones(num_img, device=attention.device, dtype=attention.dtype)

        # Text -> image attention: [B, N_important, N_vis]
        scores = attn[:, abs_text_idx, img_start:img_end]

        # Average across important text tokens: [B, N_vis]
        scores = scores.mean(dim=1)

        # Remove batch dimension (batch_size=1)
        return scores.squeeze(0)

    def post_prune(
        self,
        hidden_states: torch.Tensor,
        pruned_token_hidden_states: torch.Tensor,
        keep_indices: torch.Tensor,
        image_token_range: Tuple[int, int],
        config: VTRConfig,
        layer_idx: int,
        **context: Any,
    ) -> torch.Tensor:
        """Merge pruned tokens via clustering and insert back into sequence.

        Replicates SparseVLMs v1.0 merge pipeline:
        1. From all pruned tokens, select top 30% by text-relevance score
        2. Compute dynamic cluster count: len(candidates) / 10 + 1
        3. Cluster via density peak clustering and insert merged tokens
        """
        if not config.token_merge:
            return hidden_states

        n_pruned = pruned_token_hidden_states.shape[1]
        if n_pruned == 0:
            return hidden_states

        # Stage 1: select top 30% of pruned tokens by text-relevance
        pruned_scores = context.get("pruned_token_scores")
        if pruned_scores is not None and len(pruned_scores) > 0:
            top30_count = int(len(pruned_scores) * 0.3) + 1
            top30_count = min(top30_count, n_pruned)
            top30_idx = pruned_scores.topk(top30_count)[1]
            merge_candidates = pruned_token_hidden_states[:, top30_idx, :]
        else:
            merge_candidates = pruned_token_hidden_states

        # Stage 2: dynamic cluster count ≈ 10% of candidates + 1
        n_candidates = merge_candidates.shape[1]
        n_clusters = int(n_candidates / 10) + 1
        n_clusters = min(n_clusters, n_candidates)
        if n_clusters <= 0:
            return hidden_states

        # Cluster and merge
        merged = cluster_and_merge(merge_candidates, n_clusters)

        # Insert merged tokens at end of visual region (before text)
        img_start, img_end = image_token_range
        pre_text = hidden_states[:, :img_end, :]
        post_text = hidden_states[:, img_end:, :]
        hidden_states = torch.cat([pre_text, merged, post_text], dim=1)

        if config.debug:
            logger.debug(
                f"SparseVLM merge layer {layer_idx}: "
                f"{n_pruned} pruned -> {n_candidates} candidates "
                f"-> {n_clusters} clusters"
            )

        return hidden_states
