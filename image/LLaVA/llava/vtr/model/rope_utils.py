"""
RoPE Utilities for VTR

Unbounded RoPE implementation to handle sparse position IDs after token pruning.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


class UnboundedLlamaRotaryEmbedding(LlamaRotaryEmbedding):
    """
    [VTR-specific] Unbounded rotary position embedding.

    Standard RoPE slices the cache based on physical seq_len.
    In VTR scenarios, physical seq_len (e.g., 150) is much smaller than the retained
    position_ids (e.g., 600). This class ignores physical seq_len and always returns
    the full available cache, preventing sparse position_ids from going out of bounds.

    Example:
        Original sequence: 622 tokens, position_ids = [0, 1, 2, ..., 621]
        After pruning: 190 tokens, position_ids = [0, 1, ..., 34, 50, 80, ..., 621] (sparse)

        Standard RoPE: seq_len=190, returns cos/sin[:190], accessing position_id=621 causes OOB
        Unbounded RoPE: ignores seq_len=190, returns full cache, accessing 621 is safe
    """
    
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [bs, num_heads, seq_len, head_dim]
            seq_len: Physical sequence length (ignored in VTR)
        
        Returns:
            cos: [cache_len, head_dim]
            sin: [cache_len, head_dim]
        """
        # 1. Get the actual max length of the current cache
        cache_len = self.cos_cached.shape[0]

        # 2. Defensive check: although we ignore seq_len, if the physical length somehow
        # exceeds the cache (extremely rare), we still need to expand.
        if seq_len is not None and seq_len > cache_len:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            cache_len = self.cos_cached.shape[0]

        # 3. Key change: return the full cache directly
        # As long as max(position_ids) < cache_len, this is safe
        return (
            self.cos_cached[:cache_len].to(dtype=x.dtype),
            self.sin_cached[:cache_len].to(dtype=x.dtype),
        )
    


