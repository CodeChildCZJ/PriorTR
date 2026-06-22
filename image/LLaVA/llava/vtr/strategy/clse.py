# CLSE Strategy — Cross-Layer Spectral Evolution token pruning
# Ported into the PriorTR VTR strategy framework (LLaVA-1.5).
#
# Reference: CLSE (ECCV 2026). Original implementation prunes visual tokens in three
# progressive stages and scores them by how their spectral (frequency-domain) content
# evolves across decoder layers, combined with text->image attention.
#
# Mapping onto the PriorTR framework:
#   - config.prune_layer = [1, 11, 21]   (K_list: layers at which pruning is applied)
#   - config.ref_layers  = [0]           (L_list: snapshot reference image features z_L)
#   - config.keep_tokens = 192/128/64    (nominal layer-averaged budget -> per-stage schedule)
# The base class handles the physical pruning (Top-K, mask / position-id / KV-cache / RoPE
# re-plumbing); this strategy only supplies the per-token score and the per-stage keep count.
from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import torch

from .base import PruningStrategy
from .registry import register_strategy

if TYPE_CHECKING:
    from ..config import VTRConfig


# --- CLSE scoring primitives (ported verbatim from the CLSE reference `tools.py`) ---

def _spatial_spectral_score_2d(x: torch.Tensor, h: int = 24, w: int = 24,
                               cutoff_ratio: float = 0.1) -> torch.Tensor:
    """Per-token high-frequency spectral score via a 2D FFT high-pass filter.

    Args:
        x: token features [B, N, C], N == h * w
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
                               attn_score: torch.Tensor,
                               grid_hw: Optional[Tuple[int, int]],
                               cutoff_ratio: float, score_type: str) -> torch.Tensor:
    """Combine spectral evolution with attention. Falls back to attention-only when
    the spatial grid is unavailable (i.e. after the first pruning stage)."""
    if score_type == "attn" or grid_hw is None:
        return attn_score
    h, w = grid_hw
    s_L = _spatial_spectral_score_2d(z_L, h, w, cutoff_ratio)
    s_Lk = _spatial_spectral_score_2d(z_Lk, h, w, cutoff_ratio)
    evo = _get_evolution_factor(s_L, s_Lk)
    if score_type == "clse_attn":
        return evo * attn_score
    if score_type == "clse":
        return evo
    raise ValueError(f"Unknown score_type: {score_type}")


# Nominal layer-averaged budget -> per-stage keep counts (matches CLSE LLaVA `token_dict`).
_TOKEN_DICT = {192: [330, 210, 62], 128: [220, 140, 41], 64: [110, 70, 20]}


@register_strategy("clse")
class CLSEStrategy(PruningStrategy):
    """CLSE (Cross-Layer Spectral Evolution) progressive pruning.

    3-stage progressive prune at ``config.prune_layer`` (default [1, 11, 21]):
        stage 0: spectral-evolution x attention (``clse_attn``) on the full 24x24 grid
        stage 1, 2: attention-only (the spatial grid is broken after stage 0)
    Reference features ``z_L`` are snapshotted at ``config.ref_layers`` (default [0]) and
    routed in through ``vtr_ctx``; the current-layer features ``z_Lk`` arrive as
    ``vtr_ctx["hidden_states"]``. The per-stage keep budget is derived from
    ``config.keep_tokens`` and overrides the base single-count selection.
    """

    GRID_H = 24
    GRID_W = 24
    CUTOFF_RATIO = 0.1
    SCORE_TYPE = "clse_attn"

    def __init__(self) -> None:
        self._cur_keep: Optional[int] = None

    @staticmethod
    def _schedule(config: "VTRConfig") -> list:
        n = config.keep_tokens
        if n in _TOKEN_DICT:
            return _TOKEN_DICT[n]
        # Fallback for arbitrary budgets (same 3-stage ratios as the CLSE reference).
        return [int(n * 1.72), int(n * 1.09), int(n * 0.32)]

    def compute_scores(self, attention, image_token_range, config, **ctx):
        img_start, img_end = image_token_range

        # Which progressive stage are we in? The framework prunes after layer K-1, so the
        # incoming layer index is one of {k-1 for k in prune_layers}; its position gives the stage.
        layer_idx = ctx.get("layer_idx")
        prune_minus1 = sorted(k - 1 for k in config.prune_layers)
        stage = prune_minus1.index(layer_idx) if layer_idx in prune_minus1 else 0

        schedule = self._schedule(config)
        self._cur_keep = schedule[min(stage, len(schedule) - 1)]

        # Attention score: mean over heads, last query token's attention to the image tokens.
        attn_score = attention.mean(dim=1)[:, -1, img_start:img_end]  # [B, num_img]

        # Stage 0 (full 24x24 grid): spectral-evolution x attention. Later stages: attention-only,
        # because pruning has broken the regular grid that the 2D FFT requires.
        if stage == 0 and ("z_ref" in ctx) and ("hidden_states" in ctx):
            z_L = ctx["z_ref"]                                      # [B, 576, C]
            z_Lk = ctx["hidden_states"][:, img_start:img_end, :]    # [B, 576, C]
            score = _calculate_evolution_score(
                z_L, z_Lk, attn_score,
                (self.GRID_H, self.GRID_W), self.CUTOFF_RATIO, self.SCORE_TYPE,
            )
        else:
            score = attn_score

        return score.squeeze(0)  # [num_img]

    def select_tokens(self, scores, num_tokens, config):
        # CLSE uses a per-stage keep budget (set in compute_scores), overriding the base
        # single keep_tokens / keep_ratio selection.
        k = self._cur_keep if self._cur_keep is not None else config.get_keep_count(num_tokens)
        k = min(int(k), num_tokens)
        if k <= 0:
            return torch.tensor([], dtype=torch.long, device=scores.device)
        return scores.topk(k).indices.sort().values
