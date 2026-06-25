# CLSE LLaVA Model (Video-LLaVA)
# CLSE needs no prior forward: the reference image features z_L are snapshotted
# inside the prunable decoder loop (at config.ref_layers), so _prepare_vtr returns
# an empty ctx just like FastV.

from typing import Tuple
import torch

from .vtr_llava import VTRLlavaForCausalLM


class CLSELlava(VTRLlavaForCausalLM):
    """
    CLSE (Cross-Layer Spectral Evolution) LLaVA model.

    Like FastV, CLSE does not need an explicit prior forward; the cross-layer
    spectral snapshot happens inside PrunableLlamaModel.forward at ref_layers,
    so no preprocessing context is required.
    """

    def _prepare_vtr(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        image_token_range: Tuple[int, int],
        **kwargs,
    ) -> dict:
        """CLSE preprocessing: no extra preparation needed, returns empty ctx."""
        return {}
