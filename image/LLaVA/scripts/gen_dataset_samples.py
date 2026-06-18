#!/usr/bin/env python3
"""
Export one PDF+PNG per dataset sample, matching the reference figure style.
Each file: dataset name (bold) / square image / Q: text / A: answer (green)

Output: figures/samples/{key}.pdf + {key}.png  (12 files each)

Usage:
    python scripts/gen_dataset_samples.py [--out-dir figures/samples]
"""

import argparse, glob, io, os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import pyarrow as pa
from PIL import Image

# ── CJK font ────────────────────────────────────────────────────────────────
_CJK_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
_CJK = None
if os.path.exists(_CJK_PATH):
    fm.fontManager.addfont(_CJK_PATH)
    _CJK = fm.FontProperties(fname=_CJK_PATH)

# ── Style ────────────────────────────────────────────────────────────────────
SERIF    = "Times New Roman"
C_TITLE  = "#1a237e"
C_Q      = "#111111"
C_A      = "#2e7d32"

FS_TITLE = 15
FS_Q     = 14
FS_A     = 15

CROP_PX  = 600

# ── Dataset config ────────────────────────────────────────────────────────────
BASE = os.path.expanduser("~/.cache/huggingface/datasets")

SAMPLES = {
    "mme":       ("mme",         888,  "Is there a dog in the picture?",            "Yes"),
    "gqa":       ("gqa",           6,  "What is the airplane flying above?",        "Ocean"),
    "pope":      ("pope",         30,  "Is there a dog in the image?",              "Yes"),
    "textvqa":   ("textvqa",    5775,  "What kind of airline is this?",             "Lufthansa"),
    "seed":      ("seed",         13,  "What is the man wearing in the image?",     "A suit and tie"),
    "vizwiz":    ("vizwiz",     8547,  "What kind of phone is this?",               "BlackBerry"),
    "sqa":       ("sqa",          14,  "What is the capital of Wyoming?",           "Cheyenne"),
    "flickr30k": ("flickr30k",   176,  "Describe the image.",                       "A dog running on a rocky beach."),
    "nocaps":    ("nocaps",    10604,  "Describe the image.",                       "A blue jay on a tree."),
    "okvqa":     ("okvqa",       337,  "Egyptians worshiped these animals?",        "Cats"),
    "mmvet":     ("mmvet",        24,  "What fruit is to the right of plums?",      "Orange"),
}

MMB_IDX  = 610
MMB_EN_Q = "Which is the main topic of the image?"
MMB_EN_A = "Coffee and Dessert"
MMB_CN_Q = "图像的主要主题是什么？"
MMB_CN_A = "咖啡和甜点"

LABEL = {
    "mme":       "MME",
    "gqa":       "GQA",
    "pope":      "POPE",
    "textvqa":   "TextVQA",
    "mmbench":   "MMBench (EN & CN)",
    "seed":      "SEEDBench",
    "vizwiz":    "VizWiz",
    "sqa":       "ScienceQA",
    "flickr30k": "Flickr30k",
    "nocaps":    "NoCaps",
    "okvqa":     "OK-VQA",
    "mmvet":     "MM-Vet",
}

# ── Data helpers ──────────────────────────────────────────────────────────────

def read_table(rel):
    arrows = sorted(glob.glob(os.path.join(BASE, rel, "**/*.arrow"), recursive=True))
    if not arrows:
        raise FileNotFoundError(rel)
    tbls = []
    for a in arrows:
        try: tbls.append(pa.ipc.open_stream(a).read_all())
        except Exception: pass
    if len(tbls) == 1: return tbls[0]
    try: return pa.concat_tables(tbls)
    except pa.lib.ArrowInvalid: return pa.concat_tables(tbls, promote_options="default")


def get_img(tbl, idx):
    raw = tbl["image"][idx].as_py()
    if raw is None: return None
    if isinstance(raw, list): raw = raw[0] if raw else None
    if isinstance(raw, dict): raw = raw.get("bytes") or raw.get("path")
    if isinstance(raw, bytes):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return None


def crop_square(img, size=CROP_PX):
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s))
    return img.resize((size, size), Image.LANCZOS)

# ── Render one cell ───────────────────────────────────────────────────────────

FIG_W    = 3.6          # inches per panel
IMG_IN   = 3.4          # square image size
NAME_IN  = 0.38         # title strip height
TXT_IN   = 0.55         # Q+A strip height (tuned so 1-line Q + 1-line A nearly fill it)
FIG_H    = NAME_IN + IMG_IN + TXT_IN   # ≈ 4.88 in


def make_figure():
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[NAME_IN, IMG_IN, TXT_IN],
        hspace=0.0,
        left=0.0, right=1.0, top=1.0, bottom=0.0,
    )
    ax_name = fig.add_subplot(gs[0])
    ax_img  = fig.add_subplot(gs[1])
    ax_txt  = fig.add_subplot(gs[2])
    return fig, ax_name, ax_img, ax_txt


def setup_name(ax, label):
    ax.axis("off")
    ax.set_facecolor("white")
    ax.text(0.5, 0.5, label,
            ha="center", va="center",
            fontfamily=SERIF, fontsize=FS_TITLE,
            fontweight="bold", color=C_TITLE,
            transform=ax.transAxes)


def setup_img(ax, img):
    ax.imshow(crop_square(img))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#aaaaaa")
        sp.set_linewidth(0.7)


def setup_qa(ax, q, a, fp_q=None, fp_a=None):
    ax.axis("off")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Q — left-aligned, single line (width=55 prevents wrapping for all current Q texts)
    q_str = textwrap.fill(f"Q: {q}", width=55)
    kw_q = dict(fontsize=FS_Q, color=C_Q, ha="left", va="top",
                transform=ax.transAxes, linespacing=1.35)
    if fp_q:
        kw_q["fontproperties"] = fp_q
    else:
        kw_q["fontfamily"] = SERIF
    ax.text(0.04, 0.94, q_str, **kw_q)

    # A — right-aligned, green bold
    kw_a = dict(fontsize=FS_A, color=C_A, fontweight="bold",
                ha="right", va="bottom",
                transform=ax.transAxes)
    if fp_a:
        kw_a["fontproperties"] = fp_a
    else:
        kw_a["fontfamily"] = SERIF
    ax.text(0.96, 0.12, f"A: {a}", **kw_a)


def setup_qa_bilingual(ax):
    """MMBench: two-column layout  EN | CN  at same height as regular cells."""
    FS_B = 11   # slightly smaller to fit two columns
    ax.axis("off")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # vertical separator
    ax.plot([0.50, 0.50], [0.04, 0.96],
            color="#dddddd", linewidth=0.6, linestyle="--",
            transform=ax.transAxes)

    # ── EN side (left half) — Times New Roman ─────────────────────────────
    ax.text(0.03, 0.94, textwrap.fill(f"Q: {MMB_EN_Q}", 26),
            fontfamily=SERIF, fontsize=FS_B, color=C_Q,
            ha="left", va="top", transform=ax.transAxes, linespacing=1.2)
    ax.text(0.47, 0.08, f"A: {MMB_EN_A}",
            fontfamily=SERIF, fontsize=FS_B + 1, color=C_A, fontweight="bold",
            ha="right", va="bottom", transform=ax.transAxes)

    # ── CN side (right half) — NotoSerifCJK ──────────────────────────────
    fp = _CJK
    kw_q = dict(fontsize=FS_B, color=C_Q, ha="left", va="top",
                transform=ax.transAxes, linespacing=1.2)
    if fp: kw_q["fontproperties"] = fp
    else:  kw_q["fontfamily"] = SERIF
    ax.text(0.53, 0.94, f"Q: {MMB_CN_Q}", **kw_q)

    kw_a = dict(fontsize=FS_B + 1, color=C_A, fontweight="bold",
                ha="right", va="bottom", transform=ax.transAxes)
    if fp: kw_a["fontproperties"] = fp
    else:  kw_a["fontfamily"] = SERIF
    ax.text(0.97, 0.08, f"A: {MMB_CN_A}", **kw_a)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figures/samples")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading datasets …")
    tbls = {
        "mme":      read_table("lmms-lab___mme"),
        "gqa":      read_table("lmms-lab___gqa/testdev_balanced_instructions"),
        "pope":     read_table("lmms-lab___pope"),
        "textvqa":  read_table("lmms-lab___textvqa"),
        "mmbench":  read_table("lmms-lab___mm_bench/en"),
        "seed":     read_table("lmms-lab___seed-bench"),
        "vizwiz":   read_table("lmms-lab___viz_wiz-vqa"),
        "sqa":      read_table("lmms-lab___science_qa"),
        "flickr30k":read_table("lmms-lab___flickr30k"),
        "nocaps":   read_table("lmms-lab___no_caps"),
        "okvqa":    read_table("lmms-lab___ok-vqa"),
        "mmvet":    read_table("lmms-lab___mm_vet"),
    }
    print("  Done.")

    # ── Generate regular samples ──────────────────────────────────────────
    for key, (tbl_key, idx, q, a) in SAMPLES.items():
        img = get_img(tbls[tbl_key], idx)
        if img is None:
            print(f"  WARNING: no image for {key}[{idx}]")
            continue

        fig, ax_name, ax_img, ax_txt = make_figure()
        setup_name(ax_name, LABEL[key])
        setup_img(ax_img, img)
        setup_qa(ax_txt, q, a)

        pdf = os.path.join(args.out_dir, f"{key}.pdf")
        png = os.path.join(args.out_dir, f"{key}.png")
        fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {key}: {pdf}")

    # ── MMBench bilingual ─────────────────────────────────────────────────
    fig_mmb, ax_n, ax_i, ax_t = make_figure()
    setup_name(ax_n, LABEL["mmbench"])
    setup_img(ax_i, get_img(tbls["mmbench"], MMB_IDX))
    setup_qa_bilingual(ax_t)

    pdf = os.path.join(args.out_dir, "mmbench.pdf")
    png = os.path.join(args.out_dir, "mmbench.png")
    fig_mmb.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig_mmb.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_mmb)
    print(f"  mmbench: {pdf}")

    print(f"\nAll 12 files saved to: {args.out_dir}/")
    print("\nSuggested LaTeX (4-col grid):")
    keys = ["mme","gqa","pope","textvqa",
            "mmbench","seed","vizwiz","sqa",
            "flickr30k","nocaps","okvqa","mmvet"]
    print(r"\begin{figure}[t]")
    print(r"  \centering")
    for i, k in enumerate(keys):
        sep = r"\\" if (i+1) % 4 == 0 and i < 11 else ""
        print(f"  \\includegraphics[width=0.24\\textwidth]{{{k}}} {sep}")
    print(r"  \caption{Overview of the 12 benchmarks used for evaluation.}")
    print(r"\end{figure}")


if __name__ == "__main__":
    main()
