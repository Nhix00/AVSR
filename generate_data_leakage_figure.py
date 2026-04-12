"""
Slide 10 — Bias #1: Data Leakage
Generates a schematic showing random split (leakage) vs. GroupKFold (no leakage).
Output: report/assets/data_leakage.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_PATH = "report/assets/data_leakage.png"

# ── palette ──────────────────────────────────────────────────────────────────
DARK_RED  = "#8B1A2B"
BG_LIGHT  = "#F5F0F0"

# 4 recordings, each with 3 augmented variants
N_REC  = 4
N_AUG  = 3          # variants per recording (original + 2 augmented)
REC_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
AUG_LABELS = ["original", "+5 dB babble", "−15 dB babble"]
AUG_ALPHA  = [1.0, 0.65, 0.35]

# ── helper ───────────────────────────────────────────────────────────────────
def draw_panel(ax, title, title_color, assignments, note, note_color):
    """
    assignments: list of N_REC×N_AUG (split_label) where
                 split_label in {'train','val','test'}
    """
    SPLIT_HATCH = {"train": "", "val": "//", "test": "xx"}
    SPLIT_EDGE  = {"train": "none", "val": "#555", "test": "#222"}

    row_h = 0.55
    col_w = 1.0
    gap   = 0.18

    for r, (rec_label, aug_row, color) in enumerate(
            zip([f"Rec {i+1}" for i in range(N_REC)], assignments, REC_COLORS)):
        y = (N_REC - 1 - r) * (row_h + gap)
        for a, (split, alpha) in enumerate(zip(aug_row, AUG_ALPHA)):
            x = a * (col_w + 0.1)
            rect = mpatches.FancyBboxPatch(
                (x, y), col_w, row_h,
                boxstyle="round,pad=0.04",
                facecolor=color, alpha=alpha,
                edgecolor=SPLIT_EDGE[split], linewidth=1.4,
                hatch=SPLIT_HATCH[split],
            )
            ax.add_patch(rect)
            ax.text(x + col_w / 2, y + row_h / 2, split,
                    ha="center", va="center", fontsize=7.5,
                    color="white" if alpha > 0.5 else "#333", fontweight="bold")
        ax.text(-0.15, y + row_h / 2, rec_label,
                ha="right", va="center", fontsize=8, color="#444")

    ax.set_xlim(-0.8, N_AUG * (col_w + 0.1) + 0.1)
    ax.set_ylim(-0.55, N_REC * (row_h + gap) + 0.2)
    ax.axis("off")

    # column headers
    for a, lbl in enumerate(AUG_LABELS):
        ax.text(a * (col_w + 0.1) + col_w / 2,
                N_REC * (row_h + gap) + 0.05,
                lbl, ha="center", va="bottom", fontsize=7.5,
                color="#555", style="italic")

    ax.set_title(title, fontsize=11, fontweight="bold", color=title_color, pad=14)
    ax.text(0.5, -0.48, note, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5,
            color=note_color, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc=BG_LIGHT, ec=note_color, lw=1))


# ── random split assignments (leakage: same rec in train AND test) ────────────
rng = np.random.default_rng(42)
random_assignments = []
for _ in range(N_REC):
    # Force at least one variant in test and one in train → classic leakage
    pool = ["train", "train", "test"]
    rng.shuffle(pool)
    random_assignments.append(pool)

# ── GroupKFold assignments (all variants of a rec stay together) ──────────────
group_labels = ["train", "train", "val", "test"]
group_assignments = [[g] * N_AUG for g in group_labels]

# ── plot ─────────────────────────────────────────────────────────────────────
fig, (ax_bad, ax_good) = plt.subplots(1, 2, figsize=(11, 3.6),
                                       gridspec_kw={"wspace": 0.55})

draw_panel(ax_bad,
           "Random Split  ✗",
           "#C44E52",
           random_assignments,
           "⚠  Augmented copies of Rec 1/3 appear in BOTH train & test",
           "#C44E52")

draw_panel(ax_good,
           "Stratified Group K-Fold  ✓",
           "#2A7A4F",
           group_assignments,
           "✓  All copies of each recording stay in ONE partition",
           "#2A7A4F")

# shared legend for split types
legend_elems = [
    mpatches.Patch(facecolor="#888", label="train",  hatch="",   edgecolor="none"),
    mpatches.Patch(facecolor="#888", label="val",    hatch="//", edgecolor="#555"),
    mpatches.Patch(facecolor="#888", label="test",   hatch="xx", edgecolor="#222"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=3,
           fontsize=8.5, framealpha=0.85, handlelength=1.4,
           bbox_to_anchor=(0.5, -0.04))

fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUTPUT_PATH}")
