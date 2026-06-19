"""VTR-enabled Qwen3-VL conditional generation model.

This module provides VTRQwen3VLForConditionalGeneration, which wraps the
original Qwen3VLForConditionalGeneration to support Visual Token Reduction
(VTR) during inference. It integrates VTRConfig, pruning strategies, and
the PrunableQwen3VLTextModel into a unified interface.

Example:
    >>> from visual_token_pruning import VTRConfig
    >>> from visual_token_pruning.model import VTRQwen3VLForConditionalGeneration
    >>>
    >>> vtr_config = VTRConfig(enabled=True, strategy="fastv", keep_ratio=0.5)
    >>> model = VTRQwen3VLForConditionalGeneration.from_pretrained_vtr(
    ...     "Qwen/Qwen3-VL-8B-Instruct",
    ...     vtr_config=vtr_config,
    ... )
"""

import logging
from typing import Dict, Optional, Tuple

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
    Qwen3VLForConditionalGeneration,
)

from ..config import VTRConfig
from ..strategy.base import VTRStrategy
from ..strategy.fastv import FastVStrategy
from ..strategy.priortr_2f import PriorTR2FStrategy
from .prior_utils import build_prior_input, compute_prior_image_token_range, extract_prior_attention
from .prunable_qwen3_vl import PrunableQwen3VLTextModel

logger = logging.getLogger(__name__)


class VTRQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3-VL model with Visual Token Reduction (VTR) support.

    This class extends Qwen3VLForConditionalGeneration to support visual token
    pruning during generation. When VTR is enabled, image tokens are selectively
    pruned based on attention-derived importance scores at specified layers.

    The model supports multiple pruning strategies:
        - PriorTR: Single-forward V-Information pruning using causal attention.
        - FastV: Uses direct attention weights for pruning decisions.
        - PriorTR-2F: Uses V-Information (two-forward) for importance estimation.
        - SparseVLM: Sparse attention pruning with optional token merging.
        - VisPruner: Importance + diversity based pruning.

    When vtr_config.enabled is False, behavior is identical to the base model.

    Args:
        config: Qwen3VL model configuration.
        vtr_config: VTR configuration. If None, defaults to VTRConfig() (disabled).

    Attributes:
        vtr_config: The VTR configuration object.
        vtr_strategy: The active pruning strategy instance.
    """

    def __init__(self, config: object, vtr_config: Optional[VTRConfig] = None):
        super().__init__(config)

        self.vtr_config = vtr_config or VTRConfig()
        self.vtr_strategy = self._create_strategy(self.vtr_config.strategy)
        self._rope_position_offset = 0

        if self.vtr_config.enabled:
            self._replace_text_model()

        logger.info(
            f"VTRQwen3VLForConditionalGeneration initialized: "
            f"enabled={self.vtr_config.enabled}, "
            f"strategy={self.vtr_config.strategy}, "
            f"keep_ratio={self.vtr_config.keep_ratio}"
        )

    def _create_strategy(self, strategy_name: str) -> VTRStrategy:
        """Create a VTR strategy instance by name.

        Args:
            strategy_name: Name of the strategy ("fastv", "priortr_2f", "sparsevlm", or "priortr").

        Returns:
            An instance of the corresponding VTRStrategy subclass.

        Raises:
            ValueError: If the strategy name is unknown.
        """
        if strategy_name == "fastv":
            return FastVStrategy()
        elif strategy_name == "priortr_2f":
            return PriorTR2FStrategy()
        elif strategy_name == "sparsevlm":
            from ..strategy.sparsevlm import SparseVLMStrategy
            return SparseVLMStrategy()
        elif strategy_name == "priortr":
            from ..strategy.priortr import PriorTRStrategy
            return PriorTRStrategy()
        elif strategy_name == "vispruner":
            from ..strategy.vispruner import VisPrunerStrategy
            return VisPrunerStrategy()
        else:
            raise ValueError(
                f"Unknown VTR strategy: '{strategy_name}'. "
                f"Must be one of: 'fastv', 'priortr_2f', 'sparsevlm', 'priortr', 'vispruner'"
            )

    def _replace_text_model(self) -> None:
        """Replace the language model class with PrunableQwen3VLTextModel.

        Changes the class of self.model.language_model in-place to
        PrunableQwen3VLTextModel, preserving all weights, buffers, and state
        exactly as-is. This avoids any precision loss from re-initialization
        or device transfers.
        """
        original_lm = self.model.language_model
        original_lm.__class__ = PrunableQwen3VLTextModel

        logger.info(
            f"Swapped language_model class to PrunableQwen3VLTextModel "
            f"(in-place, no weight copy)"
        )

    def _compute_image_token_range(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
    ) -> Optional[Tuple[int, int]]:
        """Compute the start and end indices of image tokens in the input.

        Locates the contiguous range of image_pad tokens within the input
        sequence. The range is computed from the image_token_id positions
        and the spatial merge size.

        Args:
            input_ids: Input token IDs with shape [batch, seq_len].
            image_grid_thw: Image grid dimensions [num_images, 3] containing
                (temporal, height, width) for each image.

        Returns:
            Tuple of (start, end) indices, or None if no image tokens found.
        """
        if image_grid_thw is None:
            return None

        image_token_id = self.config.image_token_id
        positions = (input_ids == image_token_id).nonzero(as_tuple=False)

        if len(positions) == 0:
            return None

        start = positions[0, 1].item()

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        num_tokens = (
            image_grid_thw.prod(dim=-1) // (spatial_merge_size ** 2)
        ).sum().item()

        return (start, start + int(num_tokens))

    def _prepare_vtr(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        image_token_range: Tuple[int, int],
        **kwargs: object,
    ) -> Dict:
        """Prepare VTR context for the generation forward pass.

        For FastV, this returns an empty context (no prior needed).
        For PriorTR-2F, this runs a prior forward pass to compute baseline
        attention scores for V-Information computation.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            pixel_values: Pixel values tensor for image encoding.
            image_grid_thw: Image grid dimensions [num_images, 3].
            image_token_range: Tuple of (start, end) for image tokens.
            **kwargs: Additional arguments, may include 'processor' for
                template-mode prior construction.

        Returns:
            Dictionary containing VTR context. For PriorTR-2F, includes
            'prior_attention' tensor with shape [num_image_tokens].
        """
        vtr_context: Dict = {"image_token_range": image_token_range}

        if self.vtr_config.strategy == "priortr_2f":
            prior_input_ids = build_prior_input(
                input_ids=input_ids,
                processor=kwargs.get("processor"),
                config=self.vtr_config,
                model_config=self.config,
            )

            prior_image_range = compute_prior_image_token_range(
                prior_input_ids=prior_input_ids,
                model_config=self.config,
            )

            prior_attention = extract_prior_attention(
                model=self,
                prior_input_ids=prior_input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                image_token_range=prior_image_range,
                prune_layer=self.vtr_config.prune_layer
                if isinstance(self.vtr_config.prune_layer, int)
                else self.vtr_config.prune_layer[0],
                config=self.vtr_config,
            )

            vtr_context["prior_attention"] = prior_attention

            logger.debug(
                f"PriorTR-2F prior computed: "
                f"prior_len={prior_input_ids.shape[1]}, "
                f"prior_image_range={prior_image_range}"
            )

        return vtr_context

    def _generate_vispruner(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Generate with VisPruner: single ViT forward + true pre-LLM pruning.

        Unlike decoder-side strategies, VisPruner prunes visual tokens BEFORE
        the LLM processes them, matching the original paper's design. This runs
        the ViT only once (for both features and attention) and removes pruned
        tokens from input_ids so no decoder layer ever sees them.

        Flow:
            1. Single ViT forward with attention extraction
            2. Per-image importance (column mean) + ToMe diversity → keep mask
            3. Prune ViT features and DeepStack features
            4. Remove pruned <|image_pad|> tokens from input_ids
            5. Select kept 3D RoPE positions from pre-computed position_ids
            6. Monkey-patch get_image_features / get_rope_index → cached
            7. Call base generate() (no decoder-side VTR)
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        assert batch_size == 1, (
            "VisPruner pre-LLM pruning supports batch_size=1"
        )

        # --- Step 1: Single ViT forward with attention ---
        with torch.no_grad():
            pv_cast = pixel_values.type(self.model.visual.dtype)
            vit_out = self.model.visual(
                pv_cast,
                grid_thw=image_grid_thw,
                output_attentions=True,
                return_dict=True,
            )

        # @check_model_inputs collects every block's output into
        # .attentions; only the last entry has the real weight list.
        attn_list = vit_out.attentions[-1]  # list of [H, N_i, N_i]
        pooler_output = vit_out.pooler_output  # [total_merged, dim]
        ds_features = vit_out.deepstack_features  # list of 3 or None

        merge_size = self.model.visual.spatial_merge_size
        merge_factor = merge_size * merge_size
        split_sizes = (
            image_grid_thw.prod(-1) // merge_factor
        ).tolist()

        per_img_feats = list(torch.split(pooler_output, split_sizes))
        per_img_ds = None
        if ds_features is not None:
            per_img_ds = [
                list(torch.split(ds, split_sizes))
                for ds in ds_features
            ]

        # --- Step 2: Per-image importance + pruning ---
        total_pruned = 0
        pruned_feats: list[torch.Tensor] = []
        pruned_ds: Optional[list[list[torch.Tensor]]] = (
            [[] for _ in ds_features] if ds_features else None
        )
        keep_masks: list[torch.Tensor] = []

        for img_i, (attn_i, feats_i) in enumerate(
            zip(attn_list, per_img_feats)
        ):
            N_m = feats_i.shape[0]

            # Column mean of self-attention → per-pre-merge-token importance
            recv = attn_i.float().mean(dim=0).mean(dim=0)  # [N_pre]
            N_pre = recv.shape[0]
            if N_pre % merge_factor == 0:
                imp = recv.reshape(-1, merge_factor).mean(dim=1)
            else:
                pad_n = merge_factor - (N_pre % merge_factor)
                padded = torch.cat(
                    [recv, torch.zeros(pad_n, device=device)]
                )
                imp = padded.reshape(-1, merge_factor).mean(dim=1)

            # Strategy computes combined importance + diversity scores
            scores = self.vtr_strategy.compute_scores(
                attention=torch.empty(0, device=device),
                image_token_range=(0, N_m),
                config=self.vtr_config,
                layer_idx=0,
                vit_importance=imp,
                vit_merged_features=feats_i,
            )

            # Determine K (tokens to keep)
            if self.vtr_config.keep_tokens is not None:
                kt = self.vtr_config.keep_tokens
                keep_n = kt if isinstance(kt, int) else kt[0]
                K = min(keep_n, N_m)
            else:
                kr = self.vtr_config.keep_ratio
                ratio = kr if isinstance(kr, float) else kr[0]
                K = max(1, int(N_m * ratio))

            _, topk_idx = scores.topk(K, sorted=False)
            topk_idx = topk_idx.sort().values  # spatial order

            pruned_feats.append(feats_i[topk_idx])
            if pruned_ds is not None:
                for ds_i in range(len(ds_features)):
                    pruned_ds[ds_i].append(
                        per_img_ds[ds_i][img_i][topk_idx]
                    )

            mask = torch.zeros(N_m, dtype=torch.bool, device=device)
            mask[topk_idx] = True
            keep_masks.append(mask)
            total_pruned += N_m - K

        # No pruning → normal path
        if total_pruned == 0:
            return Qwen3VLForConditionalGeneration.generate(
                self,
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs,
            )

        logger.debug(
            "VisPruner pre-LLM: pruned %d tokens (%d → %d)",
            total_pruned,
            sum(split_sizes),
            sum(split_sizes) - total_pruned,
        )

        # --- Step 3: Build cached output ---
        cached_output = BaseModelOutputWithDeepstackFeatures(
            pooler_output=tuple(pruned_feats),
            deepstack_features=[
                torch.cat(pruned_ds[i]) for i in range(len(ds_features))
            ] if pruned_ds else None,
        )

        # --- Step 4: Remove pruned image_pad tokens from input_ids ---
        image_token_id = self.config.image_token_id
        ids = input_ids[0]
        img_pad_pos = (ids == image_token_id).nonzero(as_tuple=True)[0]

        keep_seq = torch.ones(
            ids.shape[0], dtype=torch.bool, device=device
        )
        cum = 0
        for n_img, mask in zip(split_sizes, keep_masks):
            positions = img_pad_pos[cum:cum + n_img]
            keep_seq[positions[~mask]] = False
            cum += n_img

        keep_idx = keep_seq.nonzero(as_tuple=True)[0]
        new_input_ids = ids[keep_idx].unsqueeze(0)

        # --- Step 5: Pre-compute position_ids from original input_ids ---
        attn_mask = kwargs.get("attention_mask")
        orig_pos, orig_deltas = self.model.get_rope_index(
            input_ids, image_grid_thw, attention_mask=attn_mask,
        )
        new_pos = orig_pos[:, :, keep_idx]
        new_deltas = orig_deltas + total_pruned

        # --- Step 6: Update attention_mask ---
        if attn_mask is not None:
            kwargs["attention_mask"] = attn_mask[:, keep_idx]

        # --- Step 7: Monkey-patch and generate ---
        orig_get_feats = self.model.get_image_features
        orig_get_rope = self.model.get_rope_index

        def _cached_get_image_features(
            pixel_values, image_grid_thw=None, **kw
        ):
            return cached_output

        def _cached_get_rope_index(
            input_ids=None, image_grid_thw=None,
            video_grid_thw=None, attention_mask=None,
        ):
            return new_pos, new_deltas

        original_len = input_ids.shape[1]
        try:
            self.model.get_image_features = _cached_get_image_features
            self.model.get_rope_index = _cached_get_rope_index

            result = Qwen3VLForConditionalGeneration.generate(
                self,
                input_ids=new_input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs,
            )
        finally:
            self.model.get_image_features = orig_get_feats
            self.model.get_rope_index = orig_get_rope

        # Re-align: replace pruned prefix with original input_ids
        # so callers can slice at original_len as usual
        generated = result[:, new_input_ids.shape[1]:]
        return torch.cat([input_ids, generated], dim=1)

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Generate tokens with optional Visual Token Reduction.

        When vtr_config.enabled is True:
            1. Checks for video input (skips VTR if video_pruning_mode="none")
            2. Computes image_token_range from input_ids
            3. Prepares VTR context (PriorTR-2F runs prior forward)
            4. Injects VTR parameters into generation kwargs
            5. Calls parent generate()

        When vtr_config.enabled is False:
            Directly calls parent generate() without modification.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            pixel_values: Pixel values for image encoding.
            image_grid_thw: Image grid dimensions [num_images, 3].
            **kwargs: Additional generation arguments.

        Returns:
            Generated token IDs tensor.
        """
        # Reset position offset for each generation call
        self._rope_position_offset = 0

        if not self.vtr_config.enabled:
            return super().generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs,
            )

        # Skip VTR for video inputs when video_pruning_mode is "none"
        if kwargs.get("pixel_values_videos") is not None:
            if self.vtr_config.video_pruning_mode == "none":
                logger.debug("Video input detected, skipping VTR (mode=none)")
                return super().generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    **kwargs,
                )

        # VisPruner: dedicated pre-LLM pruning path (single ViT forward)
        if self.vtr_config.strategy == "vispruner":
            if pixel_values is not None and image_grid_thw is not None:
                return self._generate_vispruner(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    **kwargs,
                )
            logger.debug("VisPruner: no image input, skipping VTR")
            return super().generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs,
            )

        # Compute image token range
        image_token_range = self._compute_image_token_range(
            input_ids, image_grid_thw
        )

        if image_token_range is None:
            logger.debug("No image tokens found, skipping VTR")
            return super().generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs,
            )

        # Prepare VTR context
        vtr_context = self._prepare_vtr(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_token_range=image_token_range,
            **kwargs,
        )

        # Inject VTR parameters
        kwargs["vtr_config"] = self.vtr_config
        kwargs["vtr_strategy"] = self.vtr_strategy
        kwargs["vtr_context"] = vtr_context
        kwargs["image_token_range"] = image_token_range

        logger.debug(
            f"VTR generate: image_token_range={image_token_range}, "
            f"strategy={self.vtr_config.strategy}"
        )

        return super().generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: object = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        is_first_iteration: bool = False,
        vtr_config: Optional[VTRConfig] = None,
        vtr_strategy: Optional[VTRStrategy] = None,
        vtr_context: Optional[Dict] = None,
        image_token_range: Optional[Tuple[int, int]] = None,
        **kwargs: object,
    ) -> Dict:
        """Prepare model inputs for generation, injecting VTR parameters.

        Extends the parent method to pass VTR parameters (config, strategy,
        context, image_token_range) into model_inputs during the first
        iteration (prefill). Subsequent iterations (decode) do not need
        VTR parameters since image tokens are only processed during prefill.

        Args:
            input_ids: Current input token IDs.
            past_key_values: Cached key/value states from previous steps.
            attention_mask: Attention mask tensor.
            inputs_embeds: Pre-computed input embeddings.
            cache_position: Cache position indices.
            position_ids: Position IDs for RoPE.
            use_cache: Whether to use KV cache.
            pixel_values: Image pixel values.
            pixel_values_videos: Video pixel values.
            image_grid_thw: Image grid dimensions.
            video_grid_thw: Video grid dimensions.
            is_first_iteration: Whether this is the prefill iteration.
            vtr_config: VTR configuration (injected by generate()).
            vtr_strategy: VTR strategy instance (injected by generate()).
            vtr_context: VTR context dict (injected by generate()).
            image_token_range: Image token range tuple (injected by generate()).
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary of model inputs for the forward pass.
        """
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # Inject VTR parameters only during the first (prefill) iteration
        if is_first_iteration and vtr_config is not None and vtr_config.enabled:
            model_inputs["vtr_config"] = vtr_config
            model_inputs["vtr_strategy"] = vtr_strategy
            model_inputs["vtr_context"] = vtr_context
            model_inputs["image_token_range"] = image_token_range

        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: object = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        **kwargs: object,
    ):
        """Forward pass with position ID correction for VTR.

        After VTR pruning reduces the KV cache during prefill, the cached
        sequence length is shorter than the original. During subsequent decode
        steps, position_ids must account for the pruned tokens to maintain
        correct RoPE positions.

        This override:
            1. During decode: computes corrected position_ids by adding the
               total number of pruned tokens to the KV cache length.
            2. After prefill: reads the total pruned tokens from vtr_context
               and stores it for future decode steps.
        """
        # During decode phase, correct position_ids if VTR pruning occurred
        if (
            position_ids is None
            and self._rope_position_offset > 0
            and past_key_values is not None
            and past_key_values.get_seq_length() > 0
            and self.model.rope_deltas is not None
        ):
            # Compute corrected position_ids, matching Qwen3VLModel.forward()
            # decode-phase logic but with the pruning offset added
            past_kv_length = past_key_values.get_seq_length()
            corrected_length = past_kv_length + self._rope_position_offset

            if inputs_embeds is not None:
                batch_size, seq_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
                device = inputs_embeds.device
            else:
                batch_size, seq_length = input_ids.shape[0], input_ids.shape[1]
                device = input_ids.device

            delta = (corrected_length + self.model.rope_deltas).to(device)
            position_ids = torch.arange(seq_length, device=device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            delta = delta.repeat_interleave(
                batch_size // delta.shape[0], dim=0
            )
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        result = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # After prefill, store the position offset from VTR pruning
        vtr_context = kwargs.get("vtr_context")
        if vtr_context is not None:
            total_pruned = vtr_context.get("total_pruned_tokens", 0)
            if total_pruned > 0:
                self._rope_position_offset = total_pruned

        return result

    @classmethod
    def from_pretrained_vtr(
        cls,
        pretrained_model_name_or_path: str,
        vtr_config: Optional[VTRConfig] = None,
        **kwargs: object,
    ) -> "VTRQwen3VLForConditionalGeneration":
        """Load a pretrained Qwen3-VL model with VTR support.

        Convenience method that loads a pretrained model and configures
        VTR components (strategy + prunable text model) after loading.
        Supports all standard HuggingFace from_pretrained arguments
        (device_map, torch_dtype, etc.).

        Args:
            pretrained_model_name_or_path: Model identifier or local path.
            vtr_config: VTR configuration. If None, uses VTRConfig() (disabled).
            **kwargs: Additional arguments passed to from_pretrained
                (e.g., device_map, torch_dtype, attn_implementation).

        Returns:
            VTRQwen3VLForConditionalGeneration instance with VTR configured.

        Example:
            >>> model = VTRQwen3VLForConditionalGeneration.from_pretrained_vtr(
            ...     "Qwen/Qwen3-VL-8B-Instruct",
            ...     vtr_config=VTRConfig(enabled=True, strategy="priortr_2f", keep_ratio=0.5),
            ...     torch_dtype=torch.float16,
            ...     device_map="auto",
            ... )
        """
        # Load model using HuggingFace standard method
        model = cls.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Configure VTR
        model.vtr_config = vtr_config or VTRConfig()
        model.vtr_strategy = model._create_strategy(model.vtr_config.strategy)

        if model.vtr_config.enabled:
            model._replace_text_model()

        logger.info(
            f"Loaded VTR model from '{pretrained_model_name_or_path}': "
            f"enabled={model.vtr_config.enabled}, "
            f"strategy={model.vtr_config.strategy}"
        )

        return model
