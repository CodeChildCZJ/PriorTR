#!/usr/bin/env python3
"""
Generate a 4×3 grid figure matching the reference style:
  - Dataset name bold above each image
  - Square image
  - Q: left-aligned below image
  - A: centered, green, bold

Output: figures/dataset_overview.pdf + figures/dataset_overview.png

Usage:
    python scripts/gen_dataset_overview.py [--out-dir figures]
"""

import argparse
import glob
import io
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import pyarrow as pa
from PIL import Image

# ---------------------------------------------------------------------------
# CJK font for MMBench-CN
# ---------------------------------------------------------------------------
_CJK_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
_CJK = None
if os.path.exists(_CJK_PATH):
    fm.fontManager.addfont(_CJK_PATH)
    _CJK = fm.FontProperties(fname=_CJK_PATH, size=8.5)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
SERIF   = "Times New Roman"
C_TITLE = "#1a237e"    # dark navy
C_Q     = "#111111"    # question text (near-black)
C_A     = "#2e7d32"    # answer: green (matches reference)

FS_TITLE = 11.0        # dataset name
FS_Q     =  9.0        # Q text
FS_A     =  9.5        # A answer (slightly larger, bold)

CROP_PX = 480

# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------
BASE = os.path.expanduser("~/.cache/huggingface/datasets")

# (table_key, index, Q_text, A_text)
# A_text will be title-cased
SAMPLES = {
    "mme":       ("mme",        888,  "Is there a dog in the picture?",             "Yes"),
    "gqa":       ("gqa",          6,  "What is the airplane flying above?",         "Ocean"),
    "pope":      ("pope",        30,  "Is there a dog in the image?",               "Yes"),
    "textvqa":   ("textvqa",   5775,  "What kind of airline is this?",              "Lufthansa"),
    "seed":      ("seed",        13,  "What is the man wearing in the image?",      "A suit and tie"),
    "vizwiz":    ("vizwiz",    8054,  "What's the name of the drink?",              "Irn-Bru"),
    "sqa":       ("sqa",         14,  "What is the capital of Wyoming?",            "Cheyenne"),
    "flickr30k": ("flickr30k",  176,  "Describe the image.",                        "A white dog running down a rocky beach."),
    "nocaps":    ("nocaps",   10604,  "Describe the image.",                        "A blue jay sitting on the branch of a tree."),
    "okvqa":     ("okvqa",        7,  "Why might someone go to this place?",        "Shopping"),
    "mmvet":     ("mmvet",       24,  "What fruit is to the right of plums?",       "Orange"),
}

# MMBench bilingual — EN + CN, same image (index 12)
MMB_IDX  = 12
MMB_EN_Q = "Which material is this spatula made of?"
MMB_EN_A = "Rubber"
MMB_CN_Q = "这个铲子是由什么材料制成的？"
MMB_CN_A = "橡胶"

LAYOUT = [
    ["mme",       "gqa",       "pope",      "textvqa"],
    ["mmbench",   "seed",      "vizwiz",    "sqa"],
    ["flickr30k", "nocaps",    "okvqa",     "mmvet"],
]

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

# ---------------------------------------------------------------------------
# Arrow helpers
# ---------------------------------------------------------------------------

def find_arrows(rel):
    return sorted(glob.glob(
        os.path.join(BASE, rel, "**/*.arrow"), recursive=True))


def read_table(rel):
    arrows = find_arrows(rel)
    if not arrows:
        raise FileNotFoundError(rel)
    tbls = []
    for a in arrows:
        try:
            tbls.append(pa.ipc.open_stream(a).read_all())
        except Exception:
            pass
    if not tbls:
        raise RuntimeError(rel)
    if len(tbls) == 1:
        return tbls[0]
    try:
        return pa.concat_tables(tbls)
    except pa.lib.ArrowInvalid:
        return pa.concat_tables(tbls, promote_options="default")


def get_img(tbl, idx):
    raw = tbl["image"][idx].as_py()
    if raw is None:
        return None
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if isinstance(raw, dict):
        raw = raw.get("bytes") or raw.get("path")
    if isinstance(raw, bytes):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return None


def crop_square(img, size=CROP_PX):
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s))
    return img.resize((size, size), Image.LANCZOS)

# ---------------------------------------------------------------------------
# Draw one normal cell
# ---------------------------------------------------------------------------

def draw_cell(ax_name, ax_img, ax_txt, img, label, q, a):
    """
    Three-row cell:
      ax_name : dataset name (small height)
      ax_img  : square image
      ax_txt  : Q + A text
    """
    # ── Name ──────────────────────────────────────────────────────────────
    ax_name.axis("off")
    ax_name.set_facecolor("white")
    ax_name.text(0.5, 0.5, label,
                 ha="center", va="center",
                 fontfamily=SERIF, fontsize=FS_TITLE,
                 fontweight="bold", color=C_TITLE,
                 transform=ax_name.transAxes)

    # ── Image ─────────────────────────────────────────────────────────────
    ax_img.imshow(crop_square(img))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_visible(True)
        sp.set_color("#bbbbbb")
        sp.set_linewidth(0.6)

    # ── Q / A text ────────────────────────────────────────────────────────
    ax_txt.axis("off")
    ax_txt.set_facecolor("white")
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)

    # Q — left-aligned, wraps at ~42 chars
    q_str = textwrap.fill(f"Q: {q}", width=42)
    ax_txt.text(0.03, 0.95, q_str,
                ha="left", va="top",
                fontfamily=SERIF, fontsize=FS_Q, color=C_Q,
                transform=ax_txt.transAxes,
                linespacing=1.3)

    # A — centered, green bold
    ax_txt.text(0.50, 0.05, f"A: {a}",
                ha="center", va="bottom",
                fontfamily=SERIF, fontsize=FS_A,
                fontweight="bold", color=C_A,
                transform=ax_txt.transAxes)


# ---------------------------------------------------------------------------
# Draw MMBench bilingual cell
# ---------------------------------------------------------------------------

def draw_mmbench_cell(ax_name, ax_img, ax_txt, img):
    # ── Name ──────────────────────────────────────────────────────────────
    ax_name.axis("off")
    ax_name.set_facecolor("white")
    ax_name.text(0.5, 0.5, LABEL["mmbench"],
                 ha="center", va="center",
                 fontfamily=SERIF, fontsize=FS_TITLE,
                 fontweight="bold", color=C_TITLE,
                 transform=ax_name.transAxes)

    # ── Image ─────────────────────────────────────────────────────────────
    ax_img.imshow(crop_square(img))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for sp in ax_img.spines.values():
        sp.set_visible(True)
        sp.set_color("#bbbbbb")
        sp.set_linewidth(0.6)

    # ── Q/A (bilingual) ───────────────────────────────────────────────────
    ax_txt.axis("off")
    ax_txt.set_facecolor("white")
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)

    fp = _CJK  # CJK font for Chinese lines

    # EN Q
    ax_txt.text(0.03, 0.97,
                textwrap.fill(f"Q: {MMB_EN_Q}", 42),
                ha="left", va="top",
                fontfamily=SERIF, fontsize=FS_Q, color=C_Q,
                transform=ax_txt.transAxes, linespacing=1.3)
    # EN A
    ax_txt.text(0.50, 0.68,
                f"A: {MMB_EN_A}",
                ha="center", va="top",
                fontfamily=SERIF, fontsize=FS_A,
                fontweight="bold", color=C_A,
                transform=ax_txt.transAxes)

    # Thin separator
    ax_txt.plot([0.03, 0.97], [0.52, 0.52],
                color="#cccccc", linewidth=0.5, linestyle="--",
                transform=ax_txt.transAxes)

    # CN Q
    ax_txt.text(0.03, 0.48,
                f"Q: {MMB_CN_Q}",
                ha="left", va="top",
                fontsize=FS_Q, color=C_Q,
                fontproperties=fp,
                transform=ax_txt.transAxes)
    # CN A
    ax_txt.text(0.50, 0.05,
                f"A: {MMB_CN_A}",
                ha="center", va="bottom",
                fontsize=FS_A, fontweight="bold", color=C_A,
                fontproperties=fp,
                transform=ax_txt.transAxes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load tables
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

    imgs = {}
    for key, (tbl_key, idx, *_) in SAMPLES.items():
        imgs[key] = get_img(tbls[tbl_key], idx)
    imgs["mmbench"] = get_img(tbls["mmbench"], MMB_IDX)

    # ── Figure layout ──────────────────────────────────────────────────────
    # Each cell: [name_row | image_row | text_row]
    # Height ratios: 0.5 : 5 : 2

    NROWS, NCOLS = 3, 4
    NAME_H, IMG_H, TXT_H = 0.5, 5, 2

    fig_w = 20.0
    col_w = fig_w / NCOLS                               # 5 in
    # total cell height (in units): NAME_H+IMG_H+TXT_H = 7.5
    # image is square (5/7.5 of cell height)
    cell_h = col_w * (NAME_H + IMG_H + TXT_H) / IMG_H  # 5 * 7.5/5 = 7.5 in
    fig_h  = NROWS * cell_h + 0.3
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    outer = gridspec.GridSpec(
        NROWS, NCOLS, figure=fig,
        hspace=0.04, wspace=0.04,
        left=0.01, right=0.99,
        top=0.99, bottom=0.01,
    )

    for ri, row_keys in enumerate(LAYOUT):
        for ci, key in enumerate(row_keys):
            inner = gridspec.GridSpecFromSubplotSpec(
                3, 1,
                subplot_spec=outer[ri, ci],
                height_ratios=[NAME_H, IMG_H, TXT_H],
                hspace=0.0,
            )
            ax_name = fig.add_subplot(inner[0])
            ax_img  = fig.add_subplot(inner[1])
            ax_txt  = fig.add_subplot(inner[2])

            if key == "mmbench":
                draw_mmbench_cell(ax_name, ax_img, ax_txt, imgs["mmbench"])
            else:
                _, idx, q, a = SAMPLES[key]
                draw_cell(ax_name, ax_img, ax_txt,
                          imgs[key], LABEL[key], q, a)

    # Save
    pdf = os.path.join(args.out_dir, "dataset_overview.pdf")
    png = os.path.join(args.out_dir, "dataset_overview.png")
    fig.savefig(pdf, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {pdf}\n  {png}")


if __name__ == "__main__":
    main()
