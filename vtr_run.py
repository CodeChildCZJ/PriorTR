#!/usr/bin/env python3
"""Unified runner for PriorTR visual-token-reduction evaluation.

One CLI to evaluate any supported (model x method) combination. Because each
base model pins a mutually-incompatible transformers version, every subproject
lives in its own conda env; this launcher does NOT run the model in-process —
it builds the correct lmms-eval command and dispatches it into the matching
env via `conda run -n <env>`. It is the generalization of how lmms-eval itself
is vendored per-env: one front-end, N isolated back-ends.

Usage:
    python vtr_run.py --list
    python vtr_run.py --model qwen3vl --method priortr --tasks mme --keep-ratio 0.2222
    python vtr_run.py --model internvl --method fastv --tasks mme --keep-tokens 192 --prune-layer 2
    python vtr_run.py --model llava --method baseline --tasks pope --dry-run

Notes:
    * lmms-eval is NOT bundled; each subproject README clones it under
      <subproject>/lmms-eval. If missing, this runner still prints the command
      (use --dry-run) but cannot execute it.
    * InfoVTR is intentionally not wired in yet (handled separately later).
"""

import argparse
import os
import shlex
import subprocess
import sys

# --------------------------------------------------------------------------- #
# Capability registry. Each entry encodes everything that differs across the
# per-env subprojects so the rest of the launcher can stay model-agnostic.
#   keys.*           : the model_args key name this subproject uses for a knob
#   fixed_args       : model_args always required by this subproject
#   baseline_args    : model_args expressing "no pruning"
#   method_extra     : extra model_args required by a specific method
#   needs_pp_parent  : export PYTHONPATH=<subproject dir> (package not pip-installed)
# NOTE: 'infovtr' is deliberately absent from every `methods` list for now.
# --------------------------------------------------------------------------- #
REGISTRY = {
    "llava": {
        "env": "PriorTRllava",
        "subdir": "image/LLaVA",
        "wrapper": "llava_vtr",
        "pretrained": "liuhaotian/llava-v1.5-7b",
        "needs_pp_parent": False,
        "fixed_args": [],
        "keys": {"strategy": "strategy", "keep_tokens": "keep_tokens",
                 "keep_ratio": "keep_ratio", "prune_layer": "prune_layer"},
        "baseline_args": ["enabled=False"],
        "methods": ["priortr", "baseline"],
        "method_extra": {},
    },
    "internvl": {
        "env": "PriorTRinternvl",
        "subdir": "image/InternVL",
        "wrapper": "internvl_vtr",
        "pretrained": "OpenGVLab/InternVL2_5-8B",
        "needs_pp_parent": True,
        "fixed_args": [],
        "keys": {"strategy": "strategy", "keep_tokens": "keep_tokens",
                 "keep_ratio": "keep_ratio", "prune_layer": "prune_layer"},
        "baseline_args": ["strategy=baseline"],
        "methods": ["priortr", "fastv", "baseline"],
        "method_extra": {},
    },
    "qwen3vl": {
        "env": "PriorTRqwen3vl",
        "subdir": "image/Qwen3-VL",
        "wrapper": "qwen3_vl_vtr",
        "pretrained": "Qwen/Qwen3-VL-8B-Instruct",
        "needs_pp_parent": False,
        "fixed_args": ["attn_implementation=sdpa"],
        "keys": {"strategy": "vtr_strategy", "keep_tokens": "vtr_keep_tokens",
                 "keep_ratio": "vtr_keep_ratio", "prune_layer": "vtr_prune_layer"},
        "baseline_args": ["vtr_enabled=False"],
        "methods": ["priortr", "fastv", "sparsevlm", "vispruner", "baseline"],
        "method_extra": {"sparsevlm": ["vtr_token_merge=True"]},
    },
}

# Methods that exist in some subprojects but are held back from the runner.
DEFERRED_METHODS = {"infovtr"}

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def print_capability_matrix():
    all_methods = ["priortr", "fastv", "sparsevlm", "vispruner", "baseline"]
    width = max(len(m) for m in REGISTRY) + 2
    header = "model".ljust(width) + "env".ljust(20) + "  ".join(all_methods)
    print(header)
    print("-" * len(header))
    for model, spec in REGISTRY.items():
        row = model.ljust(width) + spec["env"].ljust(20)
        cells = []
        for m in all_methods:
            mark = " ✓ " if m in spec["methods"] else " · "
            cells.append(mark.center(len(m)))
        print(row + "".join(cells))
    print("\n(InfoVTR is intentionally not included yet — handled separately later.)")


def build_model_args(spec, method, args):
    keys = spec["keys"]
    pretrained = args.pretrained or spec["pretrained"]
    out = [f"pretrained={pretrained}"] + list(spec["fixed_args"])

    if method == "baseline":
        out += list(spec["baseline_args"])
    else:
        out.append(f"{keys['strategy']}={method}")
        out += list(spec["method_extra"].get(method, []))
        if args.keep_tokens is not None:
            out.append(f"{keys['keep_tokens']}={args.keep_tokens}")
        elif args.keep_ratio is not None:
            out.append(f"{keys['keep_ratio']}={args.keep_ratio}")
        if args.prune_layer is not None:
            out.append(f"{keys['prune_layer']}={args.prune_layer}")

    if args.extra:
        out += [kv.strip() for kv in args.extra.split(",") if kv.strip()]
    return ",".join(out)


def default_output(model, method, args):
    slug = args.tasks.replace(",", "-")
    if method == "baseline":
        tag = "baseline"
    elif args.keep_tokens is not None:
        tag = f"{method}_k{args.keep_tokens}"
    elif args.keep_ratio is not None:
        tag = f"{method}_r{args.keep_ratio}"
    else:
        tag = method
    return f"../eval_results/{model}_{tag}_{slug}"


def build_inner_command(model, spec, method, args):
    lmms_dir = os.path.join(REPO_ROOT, spec["subdir"], "lmms-eval")
    model_args = build_model_args(spec, method, args)
    output = args.output or default_output(model, method, args)

    if args.num_processes > 1:
        launch = (f"accelerate launch --num_processes={args.num_processes} "
                  f"--main_process_port={args.port} -m lmms_eval")
    else:
        launch = "python -m lmms_eval"
    cuda = f"CUDA_VISIBLE_DEVICES={args.gpus} " if args.gpus else ""

    run = (f'{cuda}{launch} --model {spec["wrapper"]} '
           f'--model_args "{model_args}" '
           f'--tasks {args.tasks} --batch_size {args.batch_size} '
           f'--output_path {shlex.quote(output)}')

    parts = [f"cd {shlex.quote(lmms_dir)}"]
    if spec["needs_pp_parent"]:
        parts.append("export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH")
    parts.append(run)
    return " && ".join(parts), lmms_dir


def main():
    p = argparse.ArgumentParser(
        description="Unified PriorTR evaluation runner (model x method -> env-routed lmms-eval).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="Print the capability matrix and exit.")
    p.add_argument("--model", choices=list(REGISTRY), help="Which base model / subproject.")
    p.add_argument("--method", help="Pruning method (see --list for what each model supports).")
    p.add_argument("--tasks", help="lmms-eval task list, comma-separated (e.g. mme,pope).")
    p.add_argument("--keep-tokens", type=int, default=None, dest="keep_tokens",
                   help="Exact #visual tokens to keep (overrides --keep-ratio).")
    p.add_argument("--keep-ratio", type=float, default=None, dest="keep_ratio",
                   help="Fraction of visual tokens to keep.")
    p.add_argument("--prune-layer", type=int, default=None, dest="prune_layer",
                   help="Layer at which to prune (subproject default if unset).")
    p.add_argument("--pretrained", default=None, help="Override the HF checkpoint.")
    p.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES value, e.g. 0 or 0,1,2.")
    p.add_argument("--num-processes", type=int, default=1, dest="num_processes",
                   help="accelerate processes for multi-GPU (1 = plain python).")
    p.add_argument("--port", type=int, default=29500, help="accelerate main_process_port.")
    p.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    p.add_argument("--output", default=None, help="Override --output_path.")
    p.add_argument("--extra", default=None,
                   help='Raw extra model_args appended verbatim, e.g. "vtr_head_aggregation=max".')
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Print the command without executing.")
    args = p.parse_args()

    if args.list:
        print_capability_matrix()
        return 0

    # ---- validation ----
    missing = [f for f in ("model", "method", "tasks") if getattr(args, f) is None]
    if missing:
        p.error("missing required: " + ", ".join("--" + m for m in missing) + " (or use --list)")

    spec = REGISTRY[args.model]
    if args.method in DEFERRED_METHODS:
        print(f"error: method '{args.method}' is intentionally not wired into the runner yet "
              f"(handled separately later).", file=sys.stderr)
        return 2
    if args.method not in spec["methods"]:
        print(f"error: model '{args.model}' does not support method '{args.method}'.",
              file=sys.stderr)
        print(f"       supported: {', '.join(spec['methods'])}", file=sys.stderr)
        return 2
    if args.keep_tokens is not None and args.keep_ratio is not None:
        p.error("pass only one of --keep-tokens / --keep-ratio")

    inner, lmms_dir = build_inner_command(args.model, spec, args.method, args)

    print(f"# model={args.model}  method={args.method}  env={spec['env']}")
    print(f"# conda run -n {spec['env']} bash -lc '{inner}'")

    if not os.path.isdir(lmms_dir):
        print(f"\nwarning: {lmms_dir} not found — clone lmms-eval there per the subproject "
              f"README before running for real.", file=sys.stderr)
        if not args.dry_run:
            return 3

    if args.dry_run:
        return 0

    cmd = ["conda", "run", "-n", spec["env"], "--no-capture-output", "bash", "-lc", inner]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
