# PriorTR LLaVA Model
# Completes V-Information pruning in a single forward pass, no prior forward needed

from typing import Tuple
import torch

from .vtr_llava import VTRLlavaForCausalLM


class PriorTRLlava(VTRLlavaForCausalLM):
    """
    PriorTR LLaVA model.

    Exploits the natural isolation of causal attention to extract
    both P (task attention) and Q (prior attention) in a single forward pass,
    eliminating the need for an additional prior forward.

    _prepare_vtr returns an empty ctx; all computation is performed by
    PriorTRStrategy.compute_scores() in PrunableLlamaModel's layer-by-layer loop.
    """

    def _prepare_vtr(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        image_token_range: Tuple[int, int],
        **kwargs,
    ) -> dict:
        """No extra preparation needed; returns empty ctx."""
        return {}
