# FastV LLaVA Model
# FastV and similar simple methods: only need image_token_range, no prior required.

from typing import Tuple
import torch

from .vtr_llava import VTRLlavaForCausalLM


class FastVLlava(VTRLlavaForCausalLM):
    """
    FastV LLaVA model.

    Simple strategies like FastV do not need prior attention;
    _prepare_vtr returns an empty ctx.
    """

    def _prepare_vtr(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        image_token_range: Tuple[int, int],
        **kwargs,
    ) -> dict:
        """FastV preprocessing: no extra preparation needed, returns empty ctx."""
        return {}
