"""VisPruner strategy: visual-cue-based token pruning.

Uses ViT self-attention for importance scoring and ToMe-style cosine
similarity dedup for diversity, all computed pre-LLM. Scores are
pre-computed from ViT attention in _prepare_vtr() and stored in
vtr_context. compute_scores() assigns high scores to important tokens,
medium to diverse, and zero to pruned — so existing select_tokens(top-K)
gives exactly the right selection.

Reference: VisPruner (ICCV 2025) — Beyond Text-Visual Attention.
"""

import logging
from typing import Any, Tuple

import torch

from ..config import VTRConfig
from .base import VTRStrategy

logger = logging.getLogger(__name__)


class VisPrunerStrategy(VTRStrategy):

    def compute_scores(
        self,
        attention: torch.Tensor,
        image_token_range: Tuple[int, int],
        config: VTRConfig,
        layer_idx: int = 0,
        **context: Any,
    ) -> torch.Tensor:
        """Compute combined importance + diversity scores.

        Ignores decoder attention; uses pre-computed ViT attention
        from vtr_context["vit_importance"] and image features from
        vtr_context["vit_merged_features"] for ToMe dedup.
        """
        img_start, img_end = image_token_range
        num_img = img_end - img_start
        device = attention.device

        vit_importance = context.get("vit_importance")
        vit_features = context.get("vit_merged_features")

        if vit_importance is None:
            logger.warning("No vit_importance in context, returning uniform scores")
            return torch.ones(num_img, device=device)

        # Ensure lengths match (may differ if multi-image)
        if vit_importance.shape[0] != num_img:
            logger.warning(
                f"vit_importance length {vit_importance.shape[0]} != "
                f"num_img {num_img}, truncating/padding"
            )
            if vit_importance.shape[0] > num_img:
                vit_importance = vit_importance[:num_img]
            else:
                pad = torch.zeros(num_img - vit_importance.shape[0], device=device)
                vit_importance = torch.cat([vit_importance, pad])

        # Determine T (total to keep) — mirror select_tokens logic
        if config.keep_tokens is not None:
            keep = config.keep_tokens if isinstance(config.keep_tokens, int) else config.keep_tokens[0]
            T = min(keep, num_img)
        else:
            ratio = config.keep_ratio if isinstance(config.keep_ratio, float) else config.keep_ratio[0]
            T = max(1, int(num_img * ratio))

        T_imp = max(1, int(T * config.important_ratio))
        T_div = T - T_imp

        # --- Importance selection ---
        sorted_indices = vit_importance.argsort(descending=True)
        important_indices = sorted_indices[:T_imp]
        residual_indices = sorted_indices[T_imp:]

        # --- Diversity selection (ToMe-style dedup) ---
        diverse_indices = self._tome_dedup(
            vit_features, residual_indices, T_div
        ) if T_div > 0 and vit_features is not None else torch.tensor([], dtype=torch.long, device=device)

        # --- Encode as scores for select_tokens ---
        scores = torch.zeros(num_img, device=device)
        scores[important_indices] = 2.0
        if diverse_indices.numel() > 0:
            scores[diverse_indices] = 1.0

        if config.debug:
            logger.debug(
                f"VisPruner: num_img={num_img}, T={T}, T_imp={T_imp}, "
                f"T_div={T_div}, kept={int((scores > 0).sum())}"
            )

        return scores

    @staticmethod
    def _tome_dedup(
        features: torch.Tensor,
        residual_indices: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        """ToMe-style iterative odd-even cosine similarity dedup.

        Args:
            features: [num_img, dim] — post-merger image features.
            residual_indices: [R] — indices of non-important tokens.
            target_count: number of diverse tokens to keep.

        Returns:
            [target_count] indices of diverse tokens.
        """
        if target_count <= 0 or residual_indices.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=features.device)

        # Normalize for cosine similarity
        features_normed = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        idx = residual_indices.clone()

        while True:
            R = idx.shape[0]
            r = min(8, R - target_count)
            if r <= 0:
                break

            tokens = features_normed[idx]  # [R, dim]
            a = tokens[::2]   # even indices: [R//2, dim]
            b = tokens[1::2]  # odd indices:  [ceil(R/2), dim]

            # Pairwise cosine similarity between even and odd groups
            sim = a @ b.T  # [R//2, ceil(R/2)]
            max_sim = sim.max(dim=-1).values  # [R//2]

            # Keep even tokens with LOWEST max-similarity (most distinct)
            distinct = max_sim.argsort()[:len(a) - r]  # [R//2 - r]

            # Update: keep selected even + all odd
            idx = torch.cat([idx[::2][distinct], idx[1::2]])

        return idx[:target_count]

    def __repr__(self) -> str:
        return "VisPrunerStrategy()"
