"""CLSE strategy implementation (Cross-Layer Spectral Evolution).

Ported into the PriorTR / Qwen3-VL VTR strategy framework.

CLSE scores visual tokens by how their spectral (frequency-domain) content evolves
across decoder layers, combined with text->image attention. Pruning is applied in
several progressive stages (``config.prune_layer``, e.g. [1, 10, 19]); the spectral
term is only meaningful at the first stage, where the kept image tokens still form a
full ``h x w`` grid, so later stages fall back to attention-only scoring.

Framework integration (this subproject already provides most of the hooks):
  - ``prepare()`` runs once before the decoder loop with the input embeddings, so we
    snapshot the reference features ``z_L`` (the CLSE ``L_list=[0]`` reference) there.
  - ``compute_scores()`` receives ``layer_idx`` (the 0-based pruning stage) and the
    shared ``vtr_context`` (carrying ``hidden_states`` = current features ``z_Lk`` and
    ``grid_hw`` = post-merge visual grid), both routed in by the prunable model.
  - ``select_tokens()`` (inherited from the base) already supports a per-stage keep
    budget via list-valued ``keep_tokens`` / ``keep_ratio`` indexed by ``layer_idx``.

Reference: CLSE (ECCV 2026). This is a cross-model port (the original ships Qwen2-VL);
numbers are not expected to be bit-identical to the original Qwen2-VL implementation.
"""

import logging
from typing import Any, Optional, Tuple

import torch

from ..config import VTRConfig
from .base import VTRStrategy

logger = logging.getLogger(__name__)


# --- CLSE scoring primitives (ported from the CLSE reference `tools.py`) ---

def _spatial_spectral_score_2d(x: torch.Tensor, h: int, w: int,
                               cutoff_ratio: float = 0.1) -> torch.Tensor:
    """Per-token high-frequency spectral score via a 2D FFT high-pass filter.

    Args:
        x: token features [B, N, C] with N == h * w.
    Returns:
        score: [B, N]
    """
    B, N, C = x.shape
    feat = x.transpose(1, 2).reshape(B, C, h, w)

    fft_2d = torch.fft.fft2(feat.float())
    fft_shift = torch.fft.fftshift(fft_2d, dim=(-2, -1))

    center_h, center_w = h // 2, w // 2
    y, x_idx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    dist = torch.sqrt((y - center_h) ** 2 + (x_idx - center_w) ** 2).to(x.device)
    mask = 1 - torch.exp(-(dist ** 2) / (2 * (min(center_h, center_w) * cutoff_ratio) ** 2))
    filtered = fft_shift * mask.unsqueeze(0).unsqueeze(0)
    filtered = torch.fft.ifftshift(filtered, dim=(-2, -1))
    high_freq = torch.fft.ifft2(filtered).abs()

    return high_freq.mean(dim=1).reshape(B, N)


def _get_evolution_factor(s_L: torch.Tensor, s_Lk: torch.Tensor,
                          temp: float = 0.1, epsilon: float = 1e-6) -> torch.Tensor:
    """Sigmoid-normalized intensity of the spectral change between layer L and L+k."""
    evo = torch.abs(s_Lk - s_L) / (s_L.mean(dim=-1, keepdim=True) + s_L + epsilon)
    evo = torch.clamp(evo, max=1)
    mean_rate = evo.mean(dim=-1, keepdim=True)
    std_rate = evo.std(dim=-1, keepdim=True) + epsilon
    norm = (evo - mean_rate) / std_rate
    return torch.sigmoid(norm / temp)


def _calculate_evolution_score(z_L: torch.Tensor, z_Lk: torch.Tensor,
                               attn_score: torch.Tensor, grid_hw: Tuple[int, int],
                               cutoff_ratio: float, score_type: str) -> torch.Tensor:
    h, w = grid_hw
    s_L = _spatial_spectral_score_2d(z_L, h, w, cutoff_ratio)
    s_Lk = _spatial_spectral_score_2d(z_Lk, h, w, cutoff_ratio)
    evo = _get_evolution_factor(s_L, s_Lk)
    if score_type == "clse_attn":
        return evo * attn_score
    if score_type == "clse":
        return evo
    raise ValueError(f"Unknown score_type: {score_type}")


class CLSEStrategy(VTRStrategy):
    """CLSE (Cross-Layer Spectral Evolution) progressive pruning for Qwen3-VL.

    Stage 0 (first prune layer): spectral-evolution x attention on the full visual grid.
    Later stages: attention-only (the grid is broken once tokens are dropped).
    """

    CUTOFF_RATIO = 0.1
    SCORE_TYPE = "clse_attn"

    def prepare(self, hidden_states: torch.Tensor, config: VTRConfig,
                context: dict) -> None:
        """Snapshot the reference image features z_L at the model input (L_list=[0])."""
        rng = context.get("image_token_range")
        if rng is not None:
            s, e = rng
            if e > s:
                context["z_ref"] = hidden_states[:, s:e, :]

    def compute_scores(self, attention, image_token_range, config,
                       layer_idx: int = 0, **context: Any) -> torch.Tensor:
        img_start, img_end = image_token_range

        # Attention score: mean over heads, last query token's attention to image tokens.
        attn_score = attention.mean(dim=1)[:, -1, img_start:img_end]  # [B, num_img]

        grid = context.get("grid_hw")
        z_ref = context.get("z_ref")
        hs = context.get("hidden_states")

        # Spectral-evolution term only at the first stage, and only when the kept image
        # tokens still form a complete h x w grid (single, unpruned image).
        use_spectral = (
            layer_idx == 0 and grid is not None and z_ref is not None and hs is not None
        )
        if use_spectral:
            z_Lk = hs[:, img_start:img_end, :]
            num_grid = grid[0] * grid[1]
            if z_ref.shape[1] == num_grid and z_Lk.shape[1] == num_grid:
                score = _calculate_evolution_score(
                    z_ref, z_Lk, attn_score, grid, self.CUTOFF_RATIO, self.SCORE_TYPE,
                )
            else:
                # Grid mismatch (e.g. multi-image input): fall back to attention-only.
                score = attn_score
        else:
            score = attn_score

        if score.dim() == 2:
            score = score.squeeze(0)  # [num_img]
        return score

    def __repr__(self) -> str:
        return "CLSEStrategy()"
