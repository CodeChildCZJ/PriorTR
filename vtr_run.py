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
    python vtr_run.py --describe qwen3vl sparsevlm
    python vtr_run.py --model qwen3vl --method priortr --tasks mme --keep-ratio 0.2222 \
        --param query_aggregation=last --param head_aggregation=max
    python vtr_run.py --model internvl --method fastv --tasks mme --keep-tokens 192 --prune-layer 2
    python vtr_run.py --model llava --method baseline --tasks pope --dry-run

Method-specific hyperparameters are passed via repeatable --param NAME=VALUE and
are validated against the chosen method (run --describe to see what each accepts).
Common knobs (--keep-tokens/--keep-ratio/--prune-layer) apply to every method.
Default *values* are intentionally left to each subproject's own config (single
source of truth) — the launcher only injects an "intended" default where it
differs from the bare config default (e.g. SparseVLM token_merge=True).

Note: InfoVTR and Video-LLaVA are intentionally not wired in yet.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

# --------------------------------------------------------------------------- #
# Capability registry. Each entry encodes everything that differs across the
# per-env subprojects so the rest of the launcher can stay model-agnostic.
#   keys.*          : model_args key name this subproject uses for a common knob
#   fixed_args      : model_args always required by this subproject
#   baseline_args   : model_args expressing "no pruning"
#   params          : method-specific hyperparameters, keyed by a unified name:
#                       key     -> this subproject's model_args key
#                       methods -> which methods the param is read by
#                       choices -> allowed values (None = free-form)
#                       help    -> one-line description
#   method_defaults : unified-name -> value, injected for a method unless the
#                     user overrides it (only where intended != bare default)
#   method_notes    : caveats surfaced by --describe / warnings
#   needs_pp_parent : export PYTHONPATH=<subproject dir> (package not pip-installed)
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
        "params": {
            "query_aggregation": {"key": "query_aggregation", "methods": ["priortr"],
                                  "choices": ["last", "question"],
                                  "help": "query attention aggregation (auto: question@1.5, last@1.6)"},
            "head_aggregation": {"key": "head_aggregation", "methods": ["priortr"],
                                 "choices": ["mean", "max"],
                                 "help": "aggregation across attention heads"},
        },
        "method_defaults": {},
        "method_notes": {},
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
        "params": {
            "query_aggregation": {"key": "query_aggregation", "methods": ["priortr", "fastv"],
                                  "choices": ["last", "question"],
                                  "help": "query attention aggregation"},
            "head_aggregation": {"key": "head_aggregation", "methods": ["priortr", "fastv"],
                                 "choices": ["mean", "max"],
                                 "help": "aggregation across attention heads"},
            "max_num": {"key": "max_num", "methods": ["priortr", "fastv", "baseline"],
                        "choices": None,
                        "help": "max image tiles for dynamic resolution (default 6)"},
        },
        "method_defaults": {},
        "method_notes": {},
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
        "params": {
            "query_aggregation": {"key": "vtr_query_aggregation", "methods": ["priortr", "fastv"],
                                  "choices": ["last", "question", "auto"],
                                  "help": "query attention aggregation"},
            "head_aggregation": {"key": "vtr_head_aggregation", "methods": ["priortr", "fastv"],
                                 "choices": ["mean", "max"],
                                 "help": "aggregation across attention heads"},
            "token_merge": {"key": "vtr_token_merge", "methods": ["sparsevlm"],
                            "choices": ["True", "False"],
                            "help": "enable post-prune token merging (SparseVLM)"},
            "important_ratio": {"key": "vtr_important_ratio", "methods": ["vispruner"],
                                "choices": None,
                                "help": "importance/diversity split ratio (VisPruner, default 0.5)"},
        },
        "method_defaults": {"sparsevlm": {"token_merge": "True"}},
        "method_notes": {
            "vispruner": "prune_layer is forced to 1 internally (pre-LLM pruning); --prune-layer is ignored.",
        },
    },
}

# Methods that exist in some subprojects but are held back from the runner.
DEFERRED_METHODS = {"infovtr"}

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ALL_METHODS = ["priortr", "fastv", "sparsevlm", "vispruner", "baseline"]
ENV_OVERRIDES_FILE = os.path.join(REPO_ROOT, "envs.json")


# --------------------------------------------------------------------------- #
# Environment resolution & preflight.
#
# The launcher dispatches into a conda env *by name* (`conda run -n <name>`).
# Those envs are NOT created here — a user provisions them once per model by
# following the subproject README. The name the launcher uses is resolved with
# this precedence so other machines don't have to match our exact names:
#     --env <NAME>            (per-invocation override; needs --model)
#   > envs.json[model]        (per-checkout override, never committed)
#   > REGISTRY[model]["env"]  (the canonical default)
# Before running for real we verify the resolved env actually exists and, if
# not, point at the README that explains how to build it.
# --------------------------------------------------------------------------- #
def load_env_overrides():
    """Read optional REPO_ROOT/envs.json: {model: env_name}. Tolerant of absence."""
    if not os.path.isfile(ENV_OVERRIDES_FILE):
        return {}
    try:
        with open(ENV_OVERRIDES_FILE) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object of model -> env name")
        return {str(k): str(v) for k, v in data.items()}
    except (ValueError, OSError) as e:
        print(f"warning: ignoring {ENV_OVERRIDES_FILE}: {e}", file=sys.stderr)
        return {}


def resolve_env(model, spec, env_flag, overrides):
    if env_flag:
        return env_flag
    if model in overrides:
        return overrides[model]
    return spec["env"]


def list_conda_envs():
    """Set of conda env names, or None if conda can't be queried."""
    try:
        out = subprocess.run(["conda", "env", "list", "--json"],
                             capture_output=True, text=True, check=True)
        data = json.loads(out.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None
    return {os.path.basename(p) for p in data.get("envs", [])}


def print_capability_matrix():
    overrides = load_env_overrides()
    envs = list_conda_envs()
    width = max(len(m) for m in REGISTRY) + 2
    envcol = max(len(resolve_env(m, s, None, overrides)) for m, s in REGISTRY.items()) + 12
    header = "model".ljust(width) + "env".ljust(envcol) + "  ".join(ALL_METHODS)
    print(header)
    print("-" * len(header))
    for model, spec in REGISTRY.items():
        envname = resolve_env(model, spec, None, overrides)
        mark = "" if envs is None else ("  ✓" if envname in envs else "  ✗ missing")
        row = model.ljust(width) + (envname + mark).ljust(envcol)
        cells = "".join((" ✓ " if m in spec["methods"] else " · ").center(len(m))
                        for m in ALL_METHODS)
        print(row + cells)
    if envs is None:
        print("\n(could not query conda envs — is conda on PATH?)")
    print("\n(InfoVTR is intentionally not included yet — handled separately later.)")
    print("Run `--describe <model> <method>` to see that combo's tunable hyperparameters.")


def method_params(spec, method):
    """Unified param names that the given method actually reads."""
    return {name: p for name, p in spec["params"].items() if method in p["methods"]}


def describe(model, method):
    if model not in REGISTRY:
        print(f"unknown model '{model}'. choices: {', '.join(REGISTRY)}", file=sys.stderr)
        return 2
    spec = REGISTRY[model]
    if method in DEFERRED_METHODS:
        print(f"method '{method}' is intentionally not wired in yet.", file=sys.stderr)
        return 2
    if method not in spec["methods"]:
        print(f"model '{model}' does not support '{method}'. supported: "
              f"{', '.join(spec['methods'])}", file=sys.stderr)
        return 2

    print(f"{model} / {method}   (env: {spec['env']}, wrapper: {spec['wrapper']})")
    print(f"  default checkpoint: {spec['pretrained']}")
    if method == "baseline":
        print("  baseline = no pruning; only preprocessing params apply.")
    else:
        print("  common knobs: --keep-tokens | --keep-ratio, --prune-layer")
    mp = method_params(spec, method)
    if mp:
        print("  --param options for this method:")
        defaults = spec["method_defaults"].get(method, {})
        for name, p in mp.items():
            ch = f" {{{('|'.join(p['choices']))}}}" if p["choices"] else ""
            dv = f"  [default injected: {defaults[name]}]" if name in defaults else ""
            print(f"    {name}={ch:<22} {p['help']}{dv}")
    else:
        print("  (no method-specific --param options)")
    note = spec["method_notes"].get(method)
    if note:
        print(f"  note: {note}")
    return 0


def validate_params(spec, method, param_pairs):
    """Returns (translated_dict unified->value, list_of_errors)."""
    allowed = method_params(spec, method)
    out, errs = {}, []
    for name, val in param_pairs:
        if name not in spec["params"]:
            errs.append(f"unknown --param '{name}' for model (valid: "
                        f"{', '.join(spec['params']) or 'none'})")
            continue
        if name not in allowed:
            who = ", ".join(spec["params"][name]["methods"])
            errs.append(f"--param '{name}' does not apply to method '{method}' "
                        f"(applies to: {who})")
            continue
        choices = spec["params"][name]["choices"]
        if choices and val not in choices:
            errs.append(f"--param {name}={val} invalid; choices: {', '.join(choices)}")
            continue
        out[name] = val
    return out, errs


def build_model_args(spec, method, args, user_params):
    keys = spec["keys"]
    pretrained = args.pretrained or spec["pretrained"]
    out = [f"pretrained={pretrained}"] + list(spec["fixed_args"])

    if method == "baseline":
        out += list(spec["baseline_args"])
    else:
        out.append(f"{keys['strategy']}={method}")
        if args.keep_tokens is not None:
            out.append(f"{keys['keep_tokens']}={args.keep_tokens}")
        elif args.keep_ratio is not None:
            out.append(f"{keys['keep_ratio']}={args.keep_ratio}")
        if args.prune_layer is not None and method != "vispruner":
            out.append(f"{keys['prune_layer']}={args.prune_layer}")

    # Inject intended per-method defaults unless the user overrode them.
    params = dict(user_params)
    for name, val in spec["method_defaults"].get(method, {}).items():
        params.setdefault(name, val)
    # Translate unified param names -> this subproject's model_args keys.
    for name, val in params.items():
        out.append(f"{spec['params'][name]['key']}={val}")

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


def build_inner_command(model, spec, method, args, user_params):
    lmms_dir = os.path.join(REPO_ROOT, spec["subdir"], "lmms-eval")
    model_args = build_model_args(spec, method, args, user_params)
    output = args.output or default_output(model, method, args)

    if args.num_processes > 1:
        launch = (f"accelerate launch --num_processes={args.num_processes} "
                  f"--main_process_port={args.port} -m lmms_eval")
    else:
        launch = "python -m lmms_eval"
    cuda = f"CUDA_VISIBLE_DEVICES={args.gpus} " if args.gpus else ""

    limit = f" --limit {args.limit}" if args.limit else ""
    run = (f'{cuda}{launch} --model {spec["wrapper"]} '
           f'--model_args "{model_args}" '
           f'--tasks {args.tasks} --batch_size {args.batch_size}{limit} '
           f'--output_path {shlex.quote(output)}')

    # An active uv/virtualenv (VIRTUAL_ENV) shadows the conda env's python on
    # PATH — and accelerate-spawned workers inherit it too. Neutralize it and
    # put the conda env (CONDA_PREFIX, set by `conda run`) first on PATH.
    parts = ['unset VIRTUAL_ENV',
             'export PATH="$CONDA_PREFIX/bin:$PATH"',
             f"cd {shlex.quote(lmms_dir)}"]
    if spec["needs_pp_parent"]:
        parts.append("export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH")
    parts.append(run)
    return " && ".join(parts), lmms_dir


def parse_param_pairs(raw_list):
    pairs, bad = [], []
    for item in raw_list or []:
        if "=" not in item:
            bad.append(item)
            continue
        name, val = item.split("=", 1)
        pairs.append((name.strip(), val.strip()))
    return pairs, bad


def main():
    p = argparse.ArgumentParser(
        description="Unified PriorTR evaluation runner (model x method -> env-routed lmms-eval).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="Print the capability matrix and exit.")
    p.add_argument("--describe", nargs=2, metavar=("MODEL", "METHOD"),
                   help="Show tunable hyperparameters for a model/method and exit.")
    p.add_argument("--model", choices=list(REGISTRY), help="Which base model / subproject.")
    p.add_argument("--method", help="Pruning method (see --list / --describe).")
    p.add_argument("--tasks", help="lmms-eval task list, comma-separated (e.g. mme,pope).")
    p.add_argument("--keep-tokens", type=int, default=None, dest="keep_tokens",
                   help="Exact #visual tokens to keep (overrides --keep-ratio).")
    p.add_argument("--keep-ratio", type=float, default=None, dest="keep_ratio",
                   help="Fraction of visual tokens to keep.")
    p.add_argument("--prune-layer", type=int, default=None, dest="prune_layer",
                   help="Layer at which to prune (subproject default if unset).")
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", dest="params",
                   help="Method-specific hyperparameter (repeatable; validated per method).")
    p.add_argument("--env", default=None,
                   help="Override the conda env name for --model (else envs.json, else the default).")
    p.add_argument("--pretrained", default=None, help="Override the HF checkpoint.")
    p.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES value, e.g. 0 or 0,1,2.")
    p.add_argument("--num-processes", type=int, default=1, dest="num_processes",
                   help="accelerate processes for multi-GPU eval throughput (1 = plain python).")
    p.add_argument("--port", type=int, default=29500, help="accelerate main_process_port.")
    p.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    p.add_argument("--limit", default=None,
                   help="lmms-eval --limit: cap #samples (int) or fraction (float), e.g. 2 for smoke tests.")
    p.add_argument("--output", default=None, help="Override --output_path.")
    p.add_argument("--extra", default=None,
                   help='Raw extra model_args appended verbatim (unvalidated escape hatch).')
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Print the command without executing.")
    args = p.parse_args()

    if args.list:
        print_capability_matrix()
        return 0
    if args.describe:
        return describe(args.describe[0], args.describe[1])

    # ---- validation ----
    missing = [f for f in ("model", "method", "tasks") if getattr(args, f) is None]
    if missing:
        p.error("missing required: " + ", ".join("--" + m for m in missing)
                + " (or use --list / --describe)")

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

    param_pairs, bad = parse_param_pairs(args.params)
    if bad:
        p.error("malformed --param (need NAME=VALUE): " + ", ".join(bad))
    user_params, perrs = validate_params(spec, args.method, param_pairs)
    if perrs:
        for e in perrs:
            print(f"error: {e}", file=sys.stderr)
        print(f"hint: run `--describe {args.model} {args.method}` to see valid params.",
              file=sys.stderr)
        return 2
    if args.prune_layer is not None and args.method == "vispruner":
        print("warning: vispruner forces prune_layer=1 internally; --prune-layer ignored.",
              file=sys.stderr)

    env = resolve_env(args.model, spec, args.env, load_env_overrides())
    inner, lmms_dir = build_inner_command(args.model, spec, args.method, args, user_params)

    print(f"# model={args.model}  method={args.method}  env={env}")
    print(f"# conda run -n {env} bash -lc '{inner}'")

    # Preflight: the env must exist (we don't create it). Point at the README.
    readme = os.path.join(spec["subdir"], "README.md")
    envs = list_conda_envs()
    if envs is None:
        print("\nwarning: could not query conda envs (is conda on PATH?); skipping env "
              "preflight.", file=sys.stderr)
    elif env not in envs:
        print(f"\nerror: conda env '{env}' not found. Create it by following {readme} "
              f"(the env name must match — or set it with --env / envs.json).",
              file=sys.stderr)
        if not args.dry_run:
            return 4

    if not os.path.isdir(lmms_dir):
        print(f"\nwarning: {lmms_dir} not found — clone lmms-eval there per the subproject "
              f"README before running for real.", file=sys.stderr)
        if not args.dry_run:
            return 3

    if args.dry_run:
        return 0

    cmd = ["conda", "run", "-n", env, "--no-capture-output", "bash", "-lc", inner]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
