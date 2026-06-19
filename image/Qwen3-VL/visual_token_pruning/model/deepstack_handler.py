"""DeepStack synchronization handler for Visual Token Pruning.

This module provides the DeepStackSyncHandler class, which manages the
coordination between visual token pruning and Qwen3-VL's DeepStack mechanism.

DeepStack injects visual features from Vision Encoder intermediate layers
(blocks 8, 16, 24) into the Text Decoder's first 3 layers (after layers 0, 1, 2).
When pruning removes visual tokens, the DeepStack embeddings and visual position
masks must be synchronized to maintain consistency.

Timing convention:
    Layer executes -> Pruning (if score layer) -> DeepStack injection

This means that if pruning occurs at the score layer (prune_layer - 1), any
DeepStack embeddings for that layer and subsequent layers must be pruned to
match the reduced set of visual tokens.
"""

import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Default number of DeepStack layers in Qwen3-VL
DEFAULT_NUM_DEEPSTACK_LAYERS = 3


class DeepStackSyncHandler:
    """Handles synchronization of DeepStack embeddings after visual token pruning.

    This class provides methods to:
    1. Determine whether a given pruning layer requires DeepStack synchronization.
    2. Prune DeepStack embeddings that haven't been injected yet.
    3. Update visual_pos_masks to reflect the pruned sequence.

    The key insight is that pruning happens BEFORE DeepStack injection within the
    same layer iteration. So if pruning occurs at layer L (the score layer), then
    deepstack_visual_embeds[L:] all need to be pruned, because they haven't been
    injected yet at that point.

    Attributes:
        num_deepstack_layers: Number of DeepStack injection layers (default 3).

    Example:
        >>> handler = DeepStackSyncHandler(num_deepstack_layers=3)
        >>> # After pruning at layer 2 (score layer for prune_layer=3):
        >>> pruned_embeds = handler.prune_deepstack_features(
        ...     deepstack_visual_embeds=embeds,
        ...     keep_img_indices=kept_indices,
        ...     current_layer_idx=2,
        ... )
    """

    def __init__(self, num_deepstack_layers: int = DEFAULT_NUM_DEEPSTACK_LAYERS) -> None:
        """Initialize the handler.

        Args:
            num_deepstack_layers: Number of DeepStack layers used in the model.
                Defaults to 3 (matching Qwen3-VL's deepstack_visual_indexes=[8,16,24]).
        """
        self.num_deepstack_layers = num_deepstack_layers

    def needs_sync(self, current_layer_idx: int) -> bool:
        """Check if pruning at the given layer requires DeepStack synchronization.

        DeepStack injection occurs at layers 0..num_deepstack_layers-1. If pruning
        happens at a layer within this range, some DeepStack embeddings haven't
        been injected yet and need to be pruned.

        Args:
            current_layer_idx: The layer index where pruning occurs (score layer).

        Returns:
            True if DeepStack synchronization is needed.
        """
        return current_layer_idx < self.num_deepstack_layers

    def prune_deepstack_features(
        self,
        deepstack_visual_embeds: List[Optional[torch.Tensor]],
        keep_img_indices: torch.Tensor,
        current_layer_idx: int,
    ) -> List[Optional[torch.Tensor]]:
        """Prune DeepStack embeddings that haven't been injected yet.

        After pruning at layer `current_layer_idx`, all DeepStack embeddings from
        index `current_layer_idx` onwards need to be pruned because they haven't
        been injected yet (injection happens after pruning in the same iteration).

        Embeddings at indices < current_layer_idx have already been injected in
        previous iterations and are left unchanged.

        Args:
            deepstack_visual_embeds: List of visual embeddings per DeepStack layer.
                Each element has shape [num_visual_tokens, hidden_dim] or None.
            keep_img_indices: Indices of kept image tokens, relative to image range.
                Shape [num_kept].
            current_layer_idx: The layer index where pruning just occurred.

        Returns:
            New list of DeepStack embeddings with future layers pruned.
        """
        num_layers = len(deepstack_visual_embeds)
        pruned: List[Optional[torch.Tensor]] = []

        for i in range(num_layers):
            embed = deepstack_visual_embeds[i]
            if i < current_layer_idx:
                # Already injected in a previous layer iteration, leave as-is
                pruned.append(embed)
            elif embed is not None and embed.numel() > 0:
                # Not yet injected, prune to match kept tokens
                device = embed.device
                indices = keep_img_indices.to(device)
                pruned_embed = embed.index_select(0, indices)
                pruned.append(pruned_embed)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "DeepStack layer %d: pruned %s -> %s",
                        i, list(embed.shape), list(pruned_embed.shape),
                    )
            else:
                # None or empty tensor
                pruned.append(embed)

        return pruned

    def update_visual_pos_masks(
        self,
        visual_pos_masks: torch.Tensor,
        keep_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Update visual_pos_masks to match the pruned sequence.

        After pruning, the sequence length changes. The visual_pos_masks must
        be updated to correctly mark which positions in the new (shorter) sequence
        correspond to visual tokens.

        Args:
            visual_pos_masks: Boolean mask of visual token positions,
                shape [batch_size, seq_len].
            keep_indices: Full sequence indices to keep (includes sys, kept img,
                and text indices). Shape [new_seq_len].

        Returns:
            Updated visual_pos_masks with shape [batch_size, new_seq_len].
        """
        return visual_pos_masks.index_select(1, keep_indices)

    def sync_after_pruning(
        self,
        deepstack_visual_embeds: Optional[List[Optional[torch.Tensor]]],
        visual_pos_masks: Optional[torch.Tensor],
        keep_img_indices: torch.Tensor,
        keep_indices: torch.Tensor,
        current_layer_idx: int,
    ) -> Tuple[Optional[List[Optional[torch.Tensor]]], Optional[torch.Tensor]]:
        """Perform full DeepStack synchronization after pruning.

        This is the main entry point that combines pruning of DeepStack features
        and updating of visual position masks.

        Args:
            deepstack_visual_embeds: List of DeepStack embeddings, or None.
            visual_pos_masks: Visual position masks [batch, seq], or None.
            keep_img_indices: Kept image token indices (relative to image range).
            keep_indices: Full sequence keep indices.
            current_layer_idx: The layer where pruning occurred.

        Returns:
            Tuple of (updated deepstack_visual_embeds, updated visual_pos_masks).
        """
        # Update visual_pos_masks
        if visual_pos_masks is not None:
            visual_pos_masks = self.update_visual_pos_masks(
                visual_pos_masks, keep_indices
            )

        # Prune DeepStack embeddings
        if deepstack_visual_embeds is not None and self.needs_sync(current_layer_idx):
            deepstack_visual_embeds = self.prune_deepstack_features(
                deepstack_visual_embeds, keep_img_indices, current_layer_idx
            )

        return deepstack_visual_embeds, visual_pos_masks
