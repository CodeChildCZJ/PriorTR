# CLSE Strategy — Cross-Layer Spectral Evolution token pruning (Video-LLaVA)
# Ported into the PriorTR VTR strategy framework from the CLSE reference `video`
# branch (videollava/model/language_model/{clse_model.py, tools.py}).
#
# Reference: CLSE (ECCV 2026). The video variant prunes visual tokens in a single
# stage and scores them by how their spectral (frequency-domain) content evolves
# across decoder layers, combined with text->image attention.
#
# Mapping onto the PriorTR framework (Video-LLaVA, 32-layer Vicuna):
#   - config.prune_layer = [3]    (K_list: layer at which pruning is applied)
#   - config.ref_layers  = [2]    (L_list: snapshot reference image features z_L)
#   - config.keep_tokens = budget (single keep count; the reference KEEP_TOKEN)
# Video-LLaVA emits 8 x 16 x 16 = 2048 visual tokens, so the spectral term uses a
# PER-FRAME 2D FFT (temporal axis folded into the batch) — sparse 8-frame video
# aliases under a 3D FFT, so cross-layer evolution carries the temporal signal.
#
# With ref_layers=[2] snapshotting the input to layer 2 (= out(L1)) and the
# framework pruning right after layer K-1=2 (= out(L2)), z_L / z_Lk / attention
# line up exactly with the reference (z_L=out(L1), z_Lk=out(L2), attn=attn(L2)).
# The base class handles the physical pruning (Top-K, mask / position-id /
# KV-cache / RoPE re-plumbing); this strategy only supplies the per-token score.
from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import torch

from .base import PruningStrategy
from .registry import register_strategy

if TYPE_CHECKING:
    from ..config import VTRConfig


# --- CLSE scoring primitives (ported from the CLSE reference `tools.py`, video branch) ---

def _spatial_spectral_score_per_frame(x: torch.Tensor, t: int = 8, h: int = 16, w: int = 16,
                                      cutoff_ratio: float = 0.1) -> torch.Tensor:
    """Per-token high-frequency spectral score via a PER-FRAME 2D FFT high-pass.

    For sparse video (e.g. 8 frames) a 3D FFT suffers temporal aliasing, so B and T
    are merged into the batch dimension and a 2D FFT is applied independently per
    frame; cross-layer evolution is left to carry the temporal change.

    Args:
        x: token features [B, N, C], N == t * h * w
    Returns:
        score: [B, N]
    """
    B, N, C = x.shape
    device = x.device

    # [B, T*H*W, C] -> [B, T, H, W, C] -> [B*T, C, H, W]
    feat_frames = x.reshape(B, t, h, w, C).permute(0, 1, 4, 2, 3).reshape(B * t, C, h, w)

    fft_2d = torch.fft.fft2(feat_frames.float(), dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft_2d, dim=(-2, -1))

    center_h, center_w = h // 2, w // 2
    y_idx, x_idx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    dist_sq = (y_idx - center_h) ** 2 + (x_idx - center_w) ** 2
    sigma = (min(h, w) // 2) * cutoff_ratio
    mask = 1 - torch.exp(-dist_sq / (2 * sigma ** 2 + 1e-6))

    filtered = fft_shift * mask.unsqueeze(0).unsqueeze(0)
    filtered = torch.fft.ifftshift(filtered, dim=(-2, -1))
    high_freq = torch.fft.ifft2(filtered, dim=(-2, -1)).abs()

    return high_freq.mean(dim=1).reshape(B, N)


def _spatial_spectral_score_3d(x: torch.Tensor, t: int = 8, h: int = 16, w: int = 16,
                               cutoff_ratio: float = 0.1) -> torch.Tensor:
    """Per-token high-frequency spectral score via a single 3D FFT.

    The CLSE reference ships this alongside the per-frame variant but does not use
    it by default (the per-frame form avoids temporal aliasing on sparse video).
    Kept for parity / ablation via ``config.clse_fft_3d``.
    """
    B, N, C = x.shape
    device = x.device

    feat_cube = x.transpose(1, 2).reshape(B, C, t, h, w)

    fft_3d = torch.fft.fftn(feat_cube.float(), dim=(-3, -2, -1))
    fft_shift = torch.fft.fftshift(fft_3d, dim=(-3, -2, -1))

    center_t, center_h, center_w = t // 2, h // 2, w // 2
    z_idx, y_idx, x_idx = torch.meshgrid(
        torch.arange(t, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    dist_sq = (z_idx - center_t) ** 2 + (y_idx - center_h) ** 2 + (x_idx - center_w) ** 2
    sigma = (min(t, h, w) // 2) * cutoff_ratio
    mask = 1 - torch.exp(-dist_sq / (2 * sigma ** 2))

    filtered = fft_shift * mask.unsqueeze(0).unsqueeze(0)
    filtered = torch.fft.ifftshift(filtered, dim=(-3, -2, -1))
    high_freq = torch.fft.ifftn(filtered, dim=(-3, -2, -1)).abs()

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
                               grid_thw: Optional[Tuple[int, int, int]],
                               cutoff_ratio: float, score_type: str,
                               temp: float = 0.1, use_3d: bool = False) -> torch.Tensor:
    """Combine spectral evolution with attention. Falls back to attention-only when
    the spatial grid is unavailable (after pruning, or a non-matching token count)."""
    if score_type == "attn" or grid_thw is None:
        return attn_score
    t, h, w = grid_thw
    spectral = _spatial_spectral_score_3d if use_3d else _spatial_spectral_score_per_frame
    s_L = spectral(z_L, t, h, w, cutoff_ratio)
    s_Lk = spectral(z_Lk, t, h, w, cutoff_ratio)
    evo = _get_evolution_factor(s_L, s_Lk, temp)
    if score_type == "clse_attn":
        return evo * attn_score
    if score_type == "clse":
        return evo
    raise ValueError(f"Unknown score_type: {score_type}")


def apply_clse_defaults(vtr_config, hf_config=None) -> None:
    """Fill in CLSE's natural defaults so one budget knob is enough.

    When ``strategy='clse'`` and ``prune_layer`` is still a scalar (the user did not
    spell out a multi-layer schedule), resolve it to the single-stage video CLSE
    schedule ``prune_layer=[3]`` and set ``ref_layers=[2]`` for the spectral snapshot.
    No-op for any other strategy; never overrides an explicit list.
    """
    if getattr(vtr_config, "strategy", None) != "clse":
        return
    if isinstance(vtr_config.prune_layer, int) and not isinstance(vtr_config.prune_layer, bool):
        vtr_config.prune_layer = [3]
        # Refresh the cached list computed in VTRConfig.__post_init__.
        if hasattr(vtr_config, "_prune_layers"):
            vtr_config._prune_layers = [3]
    if hasattr(vtr_config, "ref_layers") and not getattr(vtr_config, "ref_layers"):
        vtr_config.ref_layers = [2]


@register_strategy("clse")
class CLSEStrategy(PruningStrategy):
    """CLSE (Cross-Layer Spectral Evolution) pruning for Video-LLaVA.

    Single-stage prune at ``config.prune_layer`` (default [3]):
        stage 0: spectral-evolution x attention (``clse_attn``) on the 8x16x16 grid
        later stages (if ever configured): attention-only (the grid is broken)
    Reference features ``z_L`` are snapshotted at ``config.ref_layers`` (default [2])
    and routed in through ``vtr_ctx``; the current-layer features ``z_Lk`` arrive as
    ``vtr_ctx["hidden_states"]``. Token selection uses the base ``keep_tokens`` count.
    """

    GRID_T = 8
    GRID_H = 16
    GRID_W = 16
    CUTOFF_RATIO = 0.1
    TEMP = 0.1
    SCORE_TYPE = "clse_attn"

    def _grid(self, config: "VTRConfig") -> Optional[Tuple[int, int, int]]:
        t = int(getattr(config, "clse_grid_t", self.GRID_T))
        h = int(getattr(config, "clse_grid_h", self.GRID_H))
        w = int(getattr(config, "clse_grid_w", self.GRID_W))
        if t <= 0 or h <= 0 or w <= 0:
            return None
        return (t, h, w)

    def compute_scores(self, attention, image_token_range, config, **ctx):
        img_start, img_end = image_token_range
        num_img = img_end - img_start

        # Which progressive stage are we in? The framework prunes after layer K-1, so
        # the incoming layer index is one of {k-1 for k in prune_layers}; its position
        # gives the stage. Video CLSE is single-stage, so this is normally just stage 0.
        layer_idx = ctx.get("layer_idx")
        prune_minus1 = sorted(k - 1 for k in config.prune_layers)
        stage = prune_minus1.index(layer_idx) if layer_idx in prune_minus1 else 0

        # Attention score: mean over heads, last query token's attention to image tokens.
        attn_score = attention.mean(dim=1)[:, -1, img_start:img_end]  # [B, num_img]

        # Stage 0 on the full grid: spectral-evolution x attention. Otherwise (later
        # stage, or token count not matching the grid) fall back to attention-only.
        grid = self._grid(config)
        grid_ok = grid is not None and (grid[0] * grid[1] * grid[2] == num_img)
        if stage == 0 and grid_ok and ("z_ref" in ctx) and ("hidden_states" in ctx):
            z_L = ctx["z_ref"]                                      # [B, N, C]
            z_Lk = ctx["hidden_states"][:, img_start:img_end, :]    # [B, N, C]
            cutoff = float(getattr(config, "clse_cutoff_ratio", self.CUTOFF_RATIO))
            temp = float(getattr(config, "clse_temp", self.TEMP))
            use_3d = bool(getattr(config, "clse_fft_3d", False))
            score = _calculate_evolution_score(
                z_L, z_Lk, attn_score, grid, cutoff, self.SCORE_TYPE, temp, use_3d,
            )
        else:
            score = attn_score

        return score.squeeze(0)  # [num_img]
