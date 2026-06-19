# PriorTR-2F LLaVA Models
# PriorTR-2F intermediate base class and two Pipeline implementations.
from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

import torch

from videollava.constants import IMAGE_TOKEN_INDEX
from .vtr_llava import VTRLlavaForCausalLM

if TYPE_CHECKING:
    from ..config import PriorTR2FConfig

from ..config import PriorTR2FConfig

logger = logging.getLogger(__name__)


class PriorTR2FBaseLlava(VTRLlavaForCausalLM):
    """
    PriorTR-2F intermediate base class.

    Contains PriorTR-2F-specific methods:
    - _build_prior_inputs(): build inputs for prior forward
    - _extract_layer_attention(): extract attention at specified layers
    """

    def _build_prior_inputs(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        prior_prompt: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build inputs for prior forward.

        Replaces the question part in original input_ids with prior_prompt.

        Args:
            input_ids: original input_ids
            images: image tensor
            prior_prompt: prior prompt text (empty string means empty prompt)

        Returns:
            prior_input_ids: replaced input_ids
            images: image tensor (unchanged)
        """
        # For LLaVA, prompt format is typically:
        # [BOS] + [SYSTEM] + [IMAGE] + [USER_QUESTION] + [ASSISTANT]
        # We need to replace USER_QUESTION with prior_prompt

        # Simplified: if prior_prompt is empty, keep up to image portion only
        # so attention reflects image's intrinsic saliency
        if prior_prompt == "":
            image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
            if image_token_mask.any():
                img_pos = image_token_mask.nonzero(as_tuple=True)[1][-1].item()  # last image token
                prior_input_ids = input_ids[:, :img_pos + 2]  # keep all image tokens + newline
            else:
                prior_input_ids = input_ids
        else:
            # With prior_prompt: need to reconstruct
            # Simplified handling; real applications may need tokenizer
            # Currently uses original input_ids (prior_prompt effect handled in strategy)
            prior_input_ids = input_ids

        return prior_input_ids, images

    def setup_vtr(self, config: PriorTR2FConfig):
        """Set up PriorTR-2F configuration. Also propagates to internal model."""
        self._vtr_config = config
        self.model.setup_vtr(config)

    def _extract_layer_attention(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        target_layers: List[int],
        image_token_range: Tuple[int, int],
    ) -> Dict[int, torch.Tensor]:
        """
        Forward and extract attention at specified layers.

        Args:
            input_ids: input token ids
            images: image tensor
            target_layers: list of layer indices to extract attention from
            image_token_range: image token range

        Returns:
            {layer_idx: attention_tensor} dictionary
        """
        # Prepare inputs_embeds
        position_ids = None
        attention_mask = None

        (
            _,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            None,
            None,
            images,
        )

        # Register hooks to extract attention
        attentions = {}
        hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                # output[1] is attention weights
                if len(output) > 1 and output[1] is not None:
                    attentions[layer_idx] = output[1].detach()
            return hook

        # Register hooks for target layers
        for layer_idx in target_layers:
            if layer_idx < len(self.model.layers):
                hook = self.model.layers[layer_idx].self_attn.register_forward_hook(
                    make_hook(layer_idx)
                )
                hooks.append(hook)

        # Forward (output_attentions=True to get attention weights)
        with torch.no_grad():
            self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Aggregate attention into per-image-token scores
        aggregated_attentions = {}
        for layer_idx, attn in attentions.items():
            # Use strategy's aggregation method
            scores = self.model._vtr_strategy._aggregate_attention(
                attn, image_token_range, self._vtr_config
            )
            # Normalize
            scores = scores / (scores.sum() + 1e-10)
            aggregated_attentions[layer_idx] = scores

        return aggregated_attentions


class FixedLayerPriorTR2F(PriorTR2FBaseLlava):
    """
    PriorTR-2F Pipeline A: Fixed pruning layer.

    Flow (2 Forwards):
    1. Prior Forward: use prior_prompt to get Q
    2. Task Forward + Prune: use task_prompt, compute S at prune_layer and prune
    """

    def _prepare_vtr(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        image_token_range: Tuple[int, int],
        **kwargs,
    ) -> dict:
        """
        Pipeline A preprocessing.

        1. Build prior inputs
        2. Extract prune_layer prior attention (Q)
        3. Return {prior_attention: Q}
        """
        config: PriorTR2FConfig = self._vtr_config

        # 1. Build prior inputs
        prior_input_ids, prior_images = self._build_prior_inputs(
            input_ids, images, config.prior_prompt
        )

        # 2. Extract prior attention
        # Multi-layer pruning support: extract prior attention for each prune layer
        prior_attentions = self._extract_layer_attention(
            prior_input_ids, prior_images,
            config.prune_layers,
            image_token_range
        )

        # Return all layers' prior attention
        return {"prior_attentions": prior_attentions}


class AdaptiveLayerPriorTR2F(PriorTR2FBaseLlava):
    """
    PriorTR-2F Pipeline B: Adaptive pruning layer.

    Flow (3 Forwards):
    1. Prior Forward: use prior_prompt to get Q for all candidate layers
    2. Task Forward 1: use task_prompt to get P for all candidate layers, select best layer
    3. Task Forward 2 + Prune: compute S at best layer and prune
    """

    def _prepare_vtr(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        image_token_range: Tuple[int, int],
        **kwargs,
    ) -> dict:
        """
        Pipeline B preprocessing.

        1. Prior Forward: extract Q for all candidate layers
        2. Task Forward: extract P for all candidate layers
        3. Select layer with largest KL(P||Q)
        4. Return {prior_attention: Q_best, prune_layer: best_layer}
        """
        config: PriorTR2FConfig = self._vtr_config
        candidate_layers = config.candidate_layers

        # 1. Build prior inputs
        prior_input_ids, prior_images = self._build_prior_inputs(
            input_ids, images, config.prior_prompt
        )

        # 2. Extract prior attention (Q) for all candidate layers
        Q_dict = self._extract_layer_attention(
            prior_input_ids, prior_images,
            candidate_layers,
            image_token_range
        )

        # 3. Extract task attention (P) for all candidate layers
        P_dict = self._extract_layer_attention(
            input_ids, images,
            candidate_layers,
            image_token_range
        )

        # 4. Select best layer
        best_layer = self._select_best_layer(Q_dict, P_dict)

        # 5. Update prune_layer in config (adaptive mode selects one best layer)
        self._vtr_config.prune_layer = best_layer
        self.model._vtr_config.prune_layer = best_layer

        logger.debug(f"AdaptiveLayerPriorTR2F selected layer {best_layer}")

        return {
            "prior_attentions": {best_layer: Q_dict[best_layer]},
            "selected_layer": best_layer,
        }

    def _select_best_layer(
        self,
        Q_dict: Dict[int, torch.Tensor],
        P_dict: Dict[int, torch.Tensor],
    ) -> int:
        """
        Select the best pruning layer.

        Criterion: layer with largest KL(P||Q).

        Args:
            Q_dict: {layer_idx: Q_attention}
            P_dict: {layer_idx: P_attention}

        Returns:
            best_layer: layer index with largest KL divergence
        """
        best_layer = None
        best_kl = -float('inf')

        for layer_idx in Q_dict.keys():
            if layer_idx not in P_dict:
                continue

            Q = Q_dict[layer_idx].float()
            P = P_dict[layer_idx].float()

            # Compute KL(P||Q) = sum(P * log(P / Q))
            eps = 1e-10
            kl = (P * torch.log((P + eps) / (Q + eps))).sum()

            assert not torch.isnan(kl), f"KL is nan for layer {layer_idx}"
            assert not torch.isinf(kl), f"KL is inf for layer {layer_idx}"

            kl = kl.item()
            if kl > best_kl:
                best_kl = kl
                best_layer = layer_idx

        return best_layer
