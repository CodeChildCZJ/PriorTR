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
                               cutoff_ratio: float, score_type: str,
                               temp: float = 0.1) -> torch.Tensor:
    h, w = grid_hw
    s_L = _spatial_spectral_score_2d(z_L, h, w, cutoff_ratio)
    s_Lk = _spatial_spectral_score_2d(z_Lk, h, w, cutoff_ratio)
    evo = _get_evolution_factor(s_L, s_Lk, temp)
    if score_type == "clse_attn":
        return evo * attn_score
    if score_type == "clse":
        return evo
    raise ValueError(f"Unknown score_type: {score_type}")


# Nominal retain ratio -> per-stage keep ratios (of the ORIGINAL visual length).
# Matches the original CLSE Qwen2-VL ``ratio_dict``; a single ``config.retain_ratio``
# (0.334 / 0.223 / 0.112) activates the whole 3-stage schedule so users never hand-
# convert per-stage ratios. (The LLaVA analog is a fixed-token ``_TOKEN_DICT`` keyed by
# ``keep_tokens``; Qwen images vary in size, so the schedule is expressed as ratios.)
_RATIO_DICT = {
    0.334: [0.57, 0.36, 0.098],
    0.223: [0.38, 0.24, 0.066],
    0.112: [0.19, 0.12, 0.034],
}

# Cross-model symmetry: the LLaVA-style headline budgets (``keep_tokens`` 192/128/64)
# map to the matching ``retain_ratio`` preset, so either knob selects the same schedule
# on any backbone (192≈1/3, 128≈2/9, 64≈1/9 of LLaVA's 576-token grid).
_TOKENS_TO_RATIO = {192: 0.334, 128: 0.223, 64: 0.112}

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
        return [1, 10, 19]
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
    out the 3-stage schedule), resolve it to the depth-aligned default for this model.
    No-op for any other strategy; never overrides an explicit list. Mirrors docs/CLSE.md.
    """
    if getattr(vtr_config, "strategy", None) != "clse":
        return
    if isinstance(vtr_config.prune_layer, int) and not isinstance(vtr_config.prune_layer, bool):
        vtr_config.prune_layer = default_prune_layers(_resolve_num_layers(hf_config))
        # Refresh any cached list form (LLaVA caches; Qwen reads it live).
        if hasattr(vtr_config, "_prune_layers"):
            vtr_config._prune_layers = sorted(vtr_config.prune_layer)
    if hasattr(vtr_config, "ref_layers") and not getattr(vtr_config, "ref_layers"):
        vtr_config.ref_layers = [0]


class CLSEStrategy(VTRStrategy):
    """CLSE (Cross-Layer Spectral Evolution) progressive pruning for Qwen2/Qwen3-VL.

    Stage 0 (first prune layer): spectral-evolution x attention on the full visual grid.
    Later stages: attention-only (the grid is broken once tokens are dropped).

    Keep budget. If ``config.retain_ratio`` is set, the hard-coded ``_RATIO_DICT``
    schedule is activated: at stage ``s`` the strategy keeps
    ``round(original_visual_len * schedule[s])`` tokens (absolute, taken from the
    original length captured in :meth:`prepare`). This reproduces the original CLSE
    per-stage budgets regardless of the model's ratio-of-current forward convention,
    so a single knob is enough. Otherwise it falls back to the base selection driven
    by ``keep_ratio`` / ``keep_tokens`` (ratio-of-current).
    """

    CUTOFF_RATIO = 0.1
    SCORE_TYPE = "clse_attn"

    def __init__(self) -> None:
        self._cur_keep: Optional[int] = None

    @staticmethod
    def _schedule(config: VTRConfig):
        """Per-stage keep ratios (of original length) for this budget, or None.

        Driven by ``retain_ratio``; a scalar ``keep_tokens`` of 192/128/64 is accepted
        as a cross-model alias for the matching preset (symmetry with LLaVA).
        """
        r = getattr(config, "retain_ratio", None)
        if r is None:
            kt = getattr(config, "keep_tokens", None)
            if isinstance(kt, int) and not isinstance(kt, bool) and kt in _TOKENS_TO_RATIO:
                r = _TOKENS_TO_RATIO[kt]
        if r is None:
            return None
        sched = _RATIO_DICT.get(round(float(r), 3))
        if sched is None:
            # Fallback for non-standard budgets: same 3-stage shape as the 0.334
            # reference (330/210/62 over 192 -> 1.72 / 1.09 / 0.32), scaled to r.
            sched = [r * 1.72, r * 1.09, r * 0.32]
        return sched

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

        # Resolve the per-stage absolute keep count from the hard-coded schedule, using
        # the ORIGINAL visual length (z_ref is snapshotted before any pruning).
        sched = self._schedule(config)
        if sched is not None:
            orig_len = z_ref.shape[1] if z_ref is not None else (img_end - img_start)
            stage = min(int(layer_idx), len(sched) - 1)
            # floor (not round) so a nominal retain_ratio reproduces the original CLSE
            # per-stage budget exactly: keep = int(original_len * schedule[stage]).
            self._cur_keep = max(1, int(orig_len * sched[stage]))
        else:
            self._cur_keep = None

        # Spectral-evolution term only at the first stage, and only when the kept image
        # tokens still form a complete h x w grid (single, unpruned image).
        use_spectral = (
            layer_idx == 0 and grid is not None and z_ref is not None and hs is not None
        )
        if use_spectral:
            z_Lk = hs[:, img_start:img_end, :]
            num_grid = grid[0] * grid[1]
            if z_ref.shape[1] == num_grid and z_Lk.shape[1] == num_grid:
                # Spectral hyper-parameters are configurable (default to the CLSE values).
                cutoff = float(getattr(config, "clse_cutoff_ratio", self.CUTOFF_RATIO))
                temp = float(getattr(config, "clse_temp", 0.1))
                score = _calculate_evolution_score(
                    z_ref, z_Lk, attn_score, grid, cutoff, self.SCORE_TYPE, temp,
                )
            else:
                # Grid mismatch (e.g. multi-image input): fall back to attention-only.
                score = attn_score
        else:
            score = attn_score

        if score.dim() == 2:
            score = score.squeeze(0)  # [num_img]
        return score

    def select_tokens(self, scores, num_tokens, config, layer_idx: int = 0):
        """Keep ``self._cur_keep`` tokens (absolute, hard-coded schedule) when
        ``retain_ratio`` is active; otherwise defer to the base ratio-of-current path."""
        if self._cur_keep is not None:
            k = min(int(self._cur_keep), int(num_tokens))
            if k <= 0:
                return torch.tensor([], dtype=torch.long, device=scores.device)
            return scores.topk(k).indices.sort().values
        return super().select_tokens(scores, num_tokens, config, layer_idx)

    def __repr__(self) -> str:
        return "CLSEStrategy()"
