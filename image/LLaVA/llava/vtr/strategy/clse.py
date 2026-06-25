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
                               cutoff_ratio: float, score_type: str,
                               temp: float = 0.1) -> torch.Tensor:
    """Combine spectral evolution with attention. Falls back to attention-only when
    the spatial grid is unavailable (i.e. after the first pruning stage)."""
    if score_type == "attn" or grid_hw is None:
        return attn_score
    h, w = grid_hw
    s_L = _spatial_spectral_score_2d(z_L, h, w, cutoff_ratio)
    s_Lk = _spatial_spectral_score_2d(z_Lk, h, w, cutoff_ratio)
    evo = _get_evolution_factor(s_L, s_Lk, temp)
    if score_type == "clse_attn":
        return evo * attn_score
    if score_type == "clse":
        return evo
    raise ValueError(f"Unknown score_type: {score_type}")


# Nominal layer-averaged budget -> per-stage keep counts (matches CLSE LLaVA `token_dict`).
_TOKEN_DICT = {192: [330, 210, 62], 128: [220, 140, 41], 64: [110, 70, 20]}

# Cross-model symmetry: a nominal retain_ratio (0.334 / 0.223 / 0.112) selects the
# matching keep_tokens preset, so the same single knob works here as on the Qwen
# backbones (192≈1/3, 128≈2/9, 64≈1/9 of the 576-token grid).
_RATIO_TO_TOKENS = {0.334: 192, 0.223: 128, 0.112: 64}

# Depth-aligned default 3-stage prune layers, used when CLSE is selected with only a
# budget knob (prune_layer left at its scalar default). Keyed by the LLM's decoder
# depth so the stages sit at ~0.36 / ~0.67 of depth on each backbone (see docs/CLSE.md);
# an unknown depth falls back to that same fraction.
_PRUNE_LAYERS_BY_DEPTH = {32: [1, 11, 21], 28: [1, 10, 19], 36: [1, 13, 24]}


def default_prune_layers(num_layers) -> list:
    """The CLSE 3-stage prune schedule for an LLM with ``num_layers`` decoder layers."""
    if num_layers in _PRUNE_LAYERS_BY_DEPTH:
        return list(_PRUNE_LAYERS_BY_DEPTH[num_layers])
    if not num_layers:
        return [1, 11, 21]
    return [1, max(1, round(num_layers * 0.36)), max(1, round(num_layers * 0.67))]


def _resolve_num_layers(hf_config) -> Optional[int]:
    for c in (hf_config, getattr(hf_config, "text_config", None),
              getattr(hf_config, "llm_config", None)):
        n = getattr(c, "num_hidden_layers", None)
        if isinstance(n, int) and n > 0:
            return n
    return None


def apply_clse_defaults(vtr_config, hf_config) -> None:
    """Fill in CLSE's natural defaults so one budget knob is enough.

    When ``strategy='clse'`` and ``prune_layer`` is still a scalar (the user did not spell
    out the 3-stage schedule), resolve it to the depth-aligned default for this model and
    set ``ref_layers=[0]`` for the spectral snapshot. No-op for any other strategy; never
    overrides an explicit list. Mirrors docs/CLSE.md.
    """
    if getattr(vtr_config, "strategy", None) != "clse":
        return
    if isinstance(vtr_config.prune_layer, int) and not isinstance(vtr_config.prune_layer, bool):
        vtr_config.prune_layer = default_prune_layers(_resolve_num_layers(hf_config))
        # Refresh LLaVA's cached list (computed in __post_init__); Qwen reads it live.
        if hasattr(vtr_config, "_prune_layers"):
            vtr_config._prune_layers = sorted(vtr_config.prune_layer)
    if hasattr(vtr_config, "ref_layers") and not getattr(vtr_config, "ref_layers"):
        vtr_config.ref_layers = [0]


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
        if n is None:
            # Cross-model alias: a nominal retain_ratio selects the keep_tokens preset.
            r = getattr(config, "retain_ratio", None)
            if r is not None:
                n = _RATIO_TO_TOKENS.get(round(float(r), 3), int(round(r * 576)))
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
            cutoff = float(getattr(config, "clse_cutoff_ratio", self.CUTOFF_RATIO))
            temp = float(getattr(config, "clse_temp", 0.1))
            score = _calculate_evolution_score(
                z_L, z_Lk, attn_score,
                (self.GRID_H, self.GRID_W), cutoff, self.SCORE_TYPE, temp,
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
