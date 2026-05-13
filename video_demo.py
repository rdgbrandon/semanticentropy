"""
video_demo.py  --  Semantic Entropy on Video with Alternating Frames
====================================================================
Demonstrates VideoHEDGE-style analysis on a synthetic video.

The video alternates between two scenes every frame:
  even frames (0,2,4,...) = SUNNY DAY
  odd  frames (1,3,5,...) = RAINY NIGHT

Three sampling strategies show how frame selection affects entropy:
  A) Even frames only  ->  all "sunny" responses  ->  LOW entropy
  B) Odd frames only   ->  all "rainy" responses  ->  LOW entropy
  C) All frames        ->  mixed responses         ->  HIGH entropy

Run:
    python video_demo.py              # DeBERTa large (paper model)
    python video_demo.py --fast       # DeBERTa small (~180 MB)
    python video_demo.py --heuristic  # offline, no downloads

Output:
    video_entropy_demo.png    -- visualization
    alternating_video.gif     -- synthetic GIF  (needs: pip install imageio Pillow)
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ask import (
    get_semantic_ids, logsumexp_by_id, semantic_entropy,
    cluster_assignment_entropy, EntailmentContent,
    EntailmentDebertaSmall, EntailmentDebertaLarge,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUESTION = "What is the weather like in the video?"
N_FRAMES = 10

CLUSTER_COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000", "#9B59B6", "#E74C3C"]
CLUSTER_LIGHT  = ["#D6E4F7", "#FDEBD0", "#E9F7E3", "#FFF9E0", "#F5EEF8", "#FDEDEC"]
LEVEL_COLORS   = {"LOW": "#27AE60", "MEDIUM": "#E67E22", "HIGH": "#C0392B"}

# ---------------------------------------------------------------------------
# Sampling strategies + simulated model responses
#
# In a real system these would come from GPT-4o / Gemini given actual frames.
# Here we simulate what a vision model would plausibly say for each strategy.
# ---------------------------------------------------------------------------

STRATEGIES = [
    {
        "label":     "Strategy A",
        "subtitle":  "Even frames only  (all SUNNY)",
        "frame_ids": [0, 2, 4, 6, 8],
        "responses": [
            "It's a bright and sunny day.",
            "The weather looks clear and sunny.",
            "It appears to be a beautiful sunny day.",
            "The sky is clear with bright sunshine.",
            "It looks sunny outside.",
            "Nice and sunny weather is visible.",
            "The scene shows clear sunny skies.",
            "It is a sunny day with no clouds.",
        ],
        "log_likelihoods": [-0.90, -0.80, -1.00, -0.85, -0.95, -0.88, -0.92, -0.87],
    },
    {
        "label":     "Strategy B",
        "subtitle":  "Odd frames only  (all RAINY)",
        "frame_ids": [1, 3, 5, 7, 9],
        "responses": [
            "It's dark and rainy outside.",
            "The weather is stormy and wet.",
            "It appears to be raining heavily.",
            "The scene shows a dark rainy night.",
            "It's raining outside.",
            "The weather looks wet and gloomy.",
            "Heavy rain is visible in the scene.",
            "It is a rainy and dark night.",
        ],
        "log_likelihoods": [-0.90, -0.80, -1.00, -0.85, -0.95, -0.88, -0.92, -0.87],
    },
    {
        "label":     "Strategy C",
        "subtitle":  "All frames  (alternating SUNNY + RAINY)",
        "frame_ids": list(range(N_FRAMES)),
        "responses": [
            "It's a bright and sunny day.",
            "It's dark and rainy outside.",
            "The weather looks clear and sunny.",
            "The weather is stormy and wet.",
            "It appears to be a beautiful sunny day.",
            "It's raining outside.",
            "The sky is clear with bright sunshine.",
            "It is a rainy and dark night.",
        ],
        "log_likelihoods": [-0.90, -0.90, -0.80, -0.80, -1.00, -1.00, -0.85, -0.85],
    },
]

# ---------------------------------------------------------------------------
# Entailment model loader
# ---------------------------------------------------------------------------

def load_model(mode: str):
    print("Loading entailment model ...")
    if mode == "heuristic":
        print("  Using content-word heuristic (offline).\n")
        return EntailmentContent()
    if mode == "fast":
        try:
            return EntailmentDebertaSmall()
        except Exception as e:
            print(f"  Small DeBERTa failed: {e}\n  Falling back to heuristic.\n")
            return EntailmentContent()
    try:
        return EntailmentDebertaLarge()
    except Exception as e:
        print(f"  Large DeBERTa failed: {e}\n  Trying small model ...")
        try:
            return EntailmentDebertaSmall()
        except Exception as e2:
            print(f"  Small DeBERTa also failed: {e2}\n  Using heuristic.\n")
            return EntailmentContent()

# ---------------------------------------------------------------------------
# Per-strategy semantic entropy computation
# ---------------------------------------------------------------------------

def run_strategy(strategy: dict, model) -> dict:
    responses = strategy["responses"]
    log_liks  = strategy["log_likelihoods"]

    sem_ids = get_semantic_ids(responses, model, question=QUESTION)
    clusters: dict[int, list] = {}
    for i, (r, sid) in enumerate(zip(responses, sem_ids)):
        clusters.setdefault(sid, []).append((i, r))

    ll_per_c      = logsumexp_by_id(sem_ids, log_liks)
    se            = semantic_entropy(ll_per_c)
    ca            = cluster_assignment_entropy(sem_ids)
    cluster_probs = np.exp(ll_per_c)
    cluster_probs = cluster_probs / cluster_probs.sum()
    max_e         = math.log(len(responses))
    frac          = se / max_e if max_e > 0 else 0
    level         = "LOW" if frac < 0.25 else ("MEDIUM" if frac < 0.60 else "HIGH")

    return dict(sem_ids=sem_ids, clusters=clusters, se=se, ca=ca,
                cluster_probs=cluster_probs, max_e=max_e, level=level)

# ---------------------------------------------------------------------------
# Optional: save GIF of the synthetic video
# ---------------------------------------------------------------------------

def save_gif(n_frames: int = N_FRAMES):
    try:
        import imageio
        from PIL import Image, ImageDraw
    except ImportError:
        print("  (skipping GIF -- run: pip install imageio Pillow)")
        return

    W, H = 200, 150
    frames = []
    for i in range(n_frames):
        img = Image.new("RGB", (W, H))
        d   = ImageDraw.Draw(img)
        if i % 2 == 0:                              # sunny
            d.rectangle([(0, 0),    (W, int(H * 0.75))], fill="#FFE566")
            d.rectangle([(0, int(H * 0.75)), (W, H)],    fill="#5DBB63")
            cx, cy, r = int(W * 0.75), int(H * 0.22), int(W * 0.10)
            d.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill="#FFB300")
        else:                                       # rainy
            d.rectangle([(0, 0),    (W, int(H * 0.75))], fill="#1A237E")
            d.rectangle([(0, int(H * 0.75)), (W, H)],    fill="#37474F")
            rng = np.random.RandomState(i)
            for _ in range(25):
                x = int(rng.uniform(5, W - 5))
                y = int(rng.uniform(20, int(H * 0.72)))
                d.line([(x, y), (x - 3, y + 12)], fill="#90CAF9", width=1)
        d.rectangle([(0, 0), (W, 17)], fill="#33333399")
        d.text((4, 2), f"F{i:02d}  {'SUNNY' if i % 2 == 0 else 'RAINY'}", fill="white")
        frames.append(np.array(img))

    path = "alternating_video.gif"
    imageio.mimsave(path, frames, fps=2, loop=0)
    print(f"  Saved GIF: {path}")

# ---------------------------------------------------------------------------
# Draw helpers
# ---------------------------------------------------------------------------

def draw_frame_strip(ax, n_frames: int, highlight_ids=None):
    """Row of stylised video frames. Dims non-selected; green border on selected."""
    ax.set_xlim(-0.5, n_frames - 0.5)
    ax.set_ylim(-0.40, 1.30)
    ax.axis("off")

    for i in range(n_frames):
        is_sunny = (i % 2 == 0)
        selected = (highlight_ids is None) or (i in highlight_ids)
        alpha    = 1.0 if selected else 0.15

        bg = "#FFE566" if is_sunny else "#1A237E"
        ax.add_patch(mpatches.FancyBboxPatch(
            (i - 0.42, 0.10), 0.84, 0.80,
            boxstyle="round,pad=0.02",
            facecolor=bg, edgecolor="white", linewidth=1.2,
            alpha=alpha, zorder=2,
        ))

        if is_sunny:
            ax.add_patch(plt.Circle((i, 0.70), 0.13,
                                    color="#FFB300", zorder=3, alpha=alpha))
            ax.plot([i - 0.30, i + 0.30], [0.28, 0.28],
                    color="#5DBB63", lw=3.5, zorder=3, alpha=alpha,
                    solid_capstyle="round")
        else:
            for dx in np.linspace(-0.18, 0.18, 5):
                ax.plot([i + dx, i + dx - 0.04], [0.72, 0.46],
                        color="#90CAF9", lw=1.5, zorder=3, alpha=alpha)
            ax.plot([i - 0.30, i + 0.30], [0.28, 0.28],
                    color="#455A64", lw=3.5, zorder=3, alpha=alpha,
                    solid_capstyle="round")

        ax.text(i, -0.12, f"F{i}", ha="center", va="center", fontsize=6.5,
                color="#444" if selected else "#BBB", fontweight="bold")

        if highlight_ids is not None and i in highlight_ids:
            ax.add_patch(mpatches.FancyBboxPatch(
                (i - 0.44, 0.08), 0.88, 0.84,
                boxstyle="round,pad=0.02",
                facecolor="none", edgecolor="#27AE60", linewidth=2.5, zorder=4,
            ))


def draw_responses(ax, strategy: dict, result: dict):
    responses = strategy["responses"]
    sem_ids   = result["sem_ids"]
    n         = len(responses)
    fsize     = max(6.0, 8.5 - (n - 5) * 0.20)
    row_h     = 0.78
    pad       = row_h / 2 - 0.01

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.axis("off")
    ax.set_title("Responses  (colored by cluster)", fontsize=8,
                 fontweight="bold", pad=4, loc="left")

    for i, (r, sid) in enumerate(zip(responses, sem_ids)):
        y     = n - 1 - i
        color = CLUSTER_COLORS[sid % len(CLUSTER_COLORS)]
        light = CLUSTER_LIGHT[sid  % len(CLUSTER_LIGHT)]
        short = (r[:62] + "...") if len(r) > 62 else r

        ax.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - pad), 1.0, row_h,
            boxstyle="round,pad=0.01",
            facecolor=light, edgecolor=color, linewidth=0.9,
        ))
        ax.text(0.036, y, f"C{sid}", fontsize=fsize, fontweight="bold",
                color="white", va="center", ha="center",
                bbox=dict(boxstyle="round,pad=0.22",
                          facecolor=color, edgecolor="none"))
        ax.text(0.09, y, f"[{i}] {short}", fontsize=fsize,
                va="center", color="#2C3E50")


def draw_gauge(ax, result: dict):
    se    = result["se"]
    max_e = result["max_e"]
    ca    = result["ca"]
    level = result["level"]
    color = LEVEL_COLORS[level]
    frac  = se / max_e if max_e > 0 else 0

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.30, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Semantic Entropy", fontsize=9, fontweight="bold", pad=6)

    theta = np.linspace(np.pi, 0, 300)
    ax.fill_between(np.cos(theta), np.zeros(300), np.sin(theta),
                    color="#EAECEE", zorder=1)
    if frac > 0:
        tf = np.linspace(np.pi, np.pi * (1 - frac), 300)
        ax.fill_between(np.cos(tf), np.zeros(300), np.sin(tf),
                        color=color, alpha=0.85, zorder=2)
    ax.plot(np.cos(theta), np.sin(theta), color="#BDC3C7", lw=1.5, zorder=3)
    ax.plot([-1, 1], [0, 0], color="#BDC3C7", lw=1.5, zorder=3)
    ax.add_patch(plt.Circle((0, 0), 0.55, color="white", zorder=4))

    angle = np.pi * (1 - frac)
    ax.annotate("", xy=(0.50 * np.cos(angle), 0.50 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=2.0),
                zorder=5)

    ax.text(0, 0.23, f"{se:.3f}", fontsize=15, fontweight="bold",
            ha="center", va="center", color=color, zorder=6)
    ax.text(0, 0.07, "entropy", fontsize=7,
            ha="center", va="center", color="#7F8C8D", zorder=6)
    ax.text(0, -0.10, level, fontsize=9, fontweight="bold",
            ha="center", va="center", color="white", zorder=7,
            bbox=dict(boxstyle="round,pad=0.40",
                      facecolor=color, edgecolor="none"))
    ax.text(0, -0.24, f"max={max_e:.2f}   ca={ca:.3f}",
            fontsize=6, ha="center", va="center", color="#95A5A6", zorder=6)
    ax.text(-1.08, -0.06, "0",          fontsize=6.5, ha="center", color="#95A5A6")
    ax.text( 1.08, -0.06, f"{max_e:.2f}", fontsize=6.5, ha="center", color="#95A5A6")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Video semantic entropy demo")
    parser.add_argument("--fast",      action="store_true",
                        help="Use small DeBERTa (~180 MB)")
    parser.add_argument("--heuristic", action="store_true",
                        help="Offline content-word heuristic, no downloads")
    args = parser.parse_args()

    mode  = "heuristic" if args.heuristic else ("fast" if args.fast else "large")
    model = load_model(mode)

    print("Running strategies ...")
    results = []
    for s in STRATEGIES:
        print(f"  {s['label']}: {s['subtitle']} ...")
        results.append(run_strategy(s, model))

    print()
    for s, r in zip(STRATEGIES, results):
        print(f"  {s['label']:12s}  clusters={r['sem_ids']}  "
              f"SE={r['se']:.4f}  [{r['level']}]")

    print("\nSaving GIF ...")
    save_gif()

    print("Building visualization ...")

    # -----------------------------------------------------------------------
    # Layout: title / full frame strip / 3 strategy rows / insight bar
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 21))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "Semantic Entropy on Video  --  How Frame Sampling Affects Uncertainty\n"
        f'Question asked to the model: "{QUESTION}"',
        fontsize=13, fontweight="bold", y=0.992, color="#2C3E50",
    )

    gs = GridSpec(
        5, 1, figure=fig,
        height_ratios=[0.50, 1, 1, 1, 0.38],
        hspace=0.55,
        left=0.04, right=0.97, top=0.955, bottom=0.03,
    )

    # Row 0 — full video strip
    ax_all = fig.add_subplot(gs[0])
    ax_all.set_title(
        f"Synthetic video: {N_FRAMES} frames  |  "
        "even = SUNNY DAY  /  odd = RAINY NIGHT",
        fontsize=10, fontweight="bold", loc="left", color="#2C3E50", pad=5,
    )
    draw_frame_strip(ax_all, N_FRAMES, highlight_ids=None)

    # Rows 1–3 — one per strategy
    strategy_colors = ["#2980B9", "#7D3C98", "#C0392B"]
    for row, (strategy, result, sc) in enumerate(
            zip(STRATEGIES, results, strategy_colors)):

        gs_row = gs[row + 1].subgridspec(
            1, 3, wspace=0.40,
            width_ratios=[1.4, 2.4, 1.0],
        )
        ax_frames = fig.add_subplot(gs_row[0])
        ax_resp   = fig.add_subplot(gs_row[1])
        ax_gauge  = fig.add_subplot(gs_row[2])

        ax_frames.set_title(
            f"[{strategy['label']}]  {strategy['subtitle']}",
            fontsize=9, fontweight="bold", loc="left", color=sc, pad=5,
        )
        draw_frame_strip(ax_frames, N_FRAMES,
                         highlight_ids=set(strategy["frame_ids"]))
        draw_responses(ax_resp, strategy, result)
        draw_gauge(ax_gauge, result)

    # Row 4 — insight summary bar
    ax_note = fig.add_subplot(gs[4])
    ax_note.axis("off")
    ax_note.set_xlim(0, 3)
    ax_note.set_ylim(0, 1)
    ax_note.text(
        1.5, 0.92,
        "Same video. Same question. Different frames sampled  ->  different entropy.",
        ha="center", fontsize=9.5, style="italic", color="#555",
    )
    for x, (s, r, sc) in enumerate(zip(STRATEGIES, results, strategy_colors)):
        lc = LEVEL_COLORS[r["level"]]
        ax_note.text(x + 0.5, 0.68, s["label"], ha="center",
                     fontsize=9, fontweight="bold", color=sc)
        ax_note.text(x + 0.5, 0.42, f"SE = {r['se']:.3f}",
                     ha="center", fontsize=13, fontweight="bold", color=lc)
        ax_note.text(x + 0.5, 0.12, r["level"] + " UNCERTAINTY",
                     ha="center", fontsize=9, fontweight="bold", color="white",
                     bbox=dict(boxstyle="round,pad=0.4",
                               facecolor=lc, edgecolor="none"))

    out = "video_entropy_demo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
    print(f"\nSaved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
