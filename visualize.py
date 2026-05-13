"""
visualize.py  --  Semantic Entropy Visualization
=================================================
Run:
    python visualize.py           # uses DeBERTa (downloads once if needed)
    python visualize.py --fast    # use small NLI model (~180 MB)
    python visualize.py --heuristic  # fully offline, no downloads

Produces: semantic_entropy_visualization.png  (also opens interactively)

Three test scenarios from low -> high uncertainty, showing:
  - Response list colored by semantic cluster
  - Cluster probability bar chart
  - Semicircular entropy gauge
  - Summary comparison at the bottom
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
    get_semantic_ids,
    logsumexp_by_id,
    semantic_entropy,
    cluster_assignment_entropy,
    EntailmentContent,
    EntailmentDebertaSmall,
    EntailmentDebertaLarge,
)

# ---------------------------------------------------------------------------
# Test scenarios  (responses + simulated sequence log-probs)
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "label":    "LOW UNCERTAINTY",
        "question": "What is the capital of France?",
        "responses": [
            "The capital of France is Paris.",
            "Paris is the capital city of France.",
            "France's capital is Paris.",
            "Paris serves as the capital of France.",
            "The answer is Paris.",
        ],
        "log_likelihoods": [-0.50, -0.60, -0.55, -0.65, -0.40],
    },
    {
        "label":    "MEDIUM UNCERTAINTY",
        "question": "Who invented the telephone?",
        "responses": [
            "Alexander Graham Bell invented the telephone.",
            "The telephone was invented by Alexander Graham Bell.",
            "Bell is widely credited with inventing the telephone.",
            "Elisha Gray also filed a patent on the same day as Bell.",
            "Antonio Meucci is credited by some as the true inventor.",
        ],
        "log_likelihoods": [-0.80, -0.90, -0.85, -2.10, -2.50],
    },
    {
        "label":    "HIGH UNCERTAINTY",
        "question": "What is the best programming language?",
        "responses": [
            "Python is the best language due to its simplicity.",
            "Rust is the best language for memory safety.",
            "JavaScript runs everywhere, making it the best choice.",
            "C++ is the best for high-performance systems programming.",
            "There is no single best language; it depends on context.",
        ],
        "log_likelihoods": [-1.20, -1.50, -1.30, -1.80, -1.60],
    },
]

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

CLUSTER_COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000", "#9B59B6", "#E74C3C"]
CLUSTER_LIGHT  = ["#D6E4F7", "#FDEBD0", "#E9F7E3", "#FFF9E0", "#F5EEF8", "#FDEDEC"]

LEVEL_COLORS = {"LOW": "#27AE60", "MEDIUM": "#E67E22", "HIGH": "#C0392B"}
LABEL_COLORS = {
    "LOW UNCERTAINTY":    "#27AE60",
    "MEDIUM UNCERTAINTY": "#E67E22",
    "HIGH UNCERTAINTY":   "#C0392B",
}

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
            m = EntailmentDebertaSmall()
            return m
        except Exception as e:
            print(f"  Small DeBERTa failed: {e}\n  Falling back to heuristic.\n")
            return EntailmentContent()
    # default: large (paper's model)
    try:
        m = EntailmentDebertaLarge()
        return m
    except Exception as e:
        print(f"  Large DeBERTa failed: {e}\n  Trying small model ...")
        try:
            m = EntailmentDebertaSmall()
            return m
        except Exception as e2:
            print(f"  Small DeBERTa also failed: {e2}\n  Falling back to heuristic.\n")
            return EntailmentContent()

# ---------------------------------------------------------------------------
# Per-scenario computation
# ---------------------------------------------------------------------------

def run_scenario(scenario: dict, model) -> dict:
    responses = scenario["responses"]
    log_liks  = scenario["log_likelihoods"]

    sem_ids  = get_semantic_ids(responses, model, question=scenario["question"])
    clusters: dict[int, list] = {}
    for i, (r, sid) in enumerate(zip(responses, sem_ids)):
        clusters.setdefault(sid, []).append((i, r))

    ll_per_c      = logsumexp_by_id(sem_ids, log_liks)
    se            = semantic_entropy(ll_per_c)
    ca            = cluster_assignment_entropy(sem_ids)
    cluster_probs = np.exp(ll_per_c)
    cluster_probs = cluster_probs / cluster_probs.sum()
    max_e         = math.log(len(responses))

    frac  = se / max_e if max_e > 0 else 0
    level = "LOW" if frac < 0.25 else ("MEDIUM" if frac < 0.60 else "HIGH")

    return dict(sem_ids=sem_ids, clusters=clusters, se=se, ca=ca,
                cluster_probs=cluster_probs, max_e=max_e, level=level)

# ---------------------------------------------------------------------------
# Panel: response list
# ---------------------------------------------------------------------------

def draw_responses(ax, scenario: dict, result: dict):
    responses = scenario["responses"]
    sem_ids   = result["sem_ids"]
    n         = len(responses)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.axis("off")

    for i, (r, sid) in enumerate(zip(responses, sem_ids)):
        y     = n - 1 - i
        color = CLUSTER_COLORS[sid % len(CLUSTER_COLORS)]
        light = CLUSTER_LIGHT[sid  % len(CLUSTER_LIGHT)]
        short = (r[:60] + "...") if len(r) > 60 else r

        # row background
        rect = mpatches.FancyBboxPatch(
            (0.0, y - 0.40), 1.0, 0.80,
            boxstyle="round,pad=0.01",
            facecolor=light, edgecolor=color, linewidth=1.2,
        )
        ax.add_patch(rect)

        # cluster badge
        ax.text(0.035, y, f"C{sid}", fontsize=8, fontweight="bold",
                color="white", va="center", ha="center",
                bbox=dict(boxstyle="round,pad=0.28",
                          facecolor=color, edgecolor="none"))

        # response text
        ax.text(0.09, y, f"[{i}] {short}", fontsize=7.8, va="center",
                color="#2C3E50")

# ---------------------------------------------------------------------------
# Panel: cluster probability bars
# ---------------------------------------------------------------------------

def draw_cluster_bars(ax, result: dict):
    probs    = result["cluster_probs"]
    clusters = result["clusters"]
    n        = len(probs)
    y_pos    = np.arange(n)
    colors   = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(n)]

    bars = ax.barh(y_pos, probs, color=colors,
                   edgecolor="white", linewidth=1.5, height=0.55)

    labels = []
    for cid in sorted(clusters.keys()):
        first = clusters[cid][0][1]
        short = (first[:24] + "...") if len(first) > 24 else first
        labels.append(f'C{cid}: "{short}"')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Probability", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Cluster Probabilities", fontsize=9, fontweight="bold", pad=6)

    for bar, prob in zip(bars, probs):
        ax.text(min(prob + 0.02, 0.90),
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", fontsize=7.5, fontweight="bold")

# ---------------------------------------------------------------------------
# Panel: semicircular entropy gauge
# ---------------------------------------------------------------------------

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

    theta_full = np.linspace(np.pi, 0, 300)

    # grey background semicircle
    ax.fill_between(np.cos(theta_full), np.zeros(300), np.sin(theta_full),
                    color="#EAECEE", zorder=1)

    # colored arc for current entropy
    if frac > 0:
        end_angle  = np.pi * (1 - frac)
        theta_fill = np.linspace(np.pi, end_angle, 300)
        ax.fill_between(np.cos(theta_fill), np.zeros(300), np.sin(theta_fill),
                        color=color, alpha=0.85, zorder=2)

    # arc outline + baseline
    ax.plot(np.cos(theta_full), np.sin(theta_full),
            color="#BDC3C7", linewidth=1.5, zorder=3)
    ax.plot([-1, 1], [0, 0], color="#BDC3C7", linewidth=1.5, zorder=3)

    # white centre disc
    ax.add_patch(plt.Circle((0, 0), 0.55, color="white", zorder=4))

    # needle
    angle = np.pi * (1 - frac)
    ax.annotate("",
                xy=(0.50 * np.cos(angle), 0.50 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=2.0),
                zorder=5)

    # entropy value
    ax.text(0, 0.23, f"{se:.3f}", fontsize=16, fontweight="bold",
            ha="center", va="center", color=color, zorder=6)
    ax.text(0, 0.07, "entropy", fontsize=7,
            ha="center", va="center", color="#7F8C8D", zorder=6)

    # level badge
    ax.text(0, -0.10, level, fontsize=9, fontweight="bold",
            ha="center", va="center", color="white", zorder=7,
            bbox=dict(boxstyle="round,pad=0.40",
                      facecolor=color, edgecolor="none"))

    # small footnote
    ax.text(0, -0.24, f"max={max_e:.2f}   ca={ca:.3f}",
            fontsize=6, ha="center", va="center",
            color="#95A5A6", zorder=6)

    # axis tick labels
    ax.text(-1.08, -0.06, "0",     fontsize=6.5, ha="center", color="#95A5A6")
    ax.text( 1.08, -0.06, f"{max_e:.2f}", fontsize=6.5, ha="center", color="#95A5A6")

# ---------------------------------------------------------------------------
# Bottom panel: summary comparison
# ---------------------------------------------------------------------------

def draw_summary(ax, results: list[dict], scenarios: list[dict]):
    labels   = [s["label"].replace(" UNCERTAINTY", "\nUNCERTAINTY")
                for s in scenarios]
    se_vals  = [r["se"]    for r in results]
    max_vals = [r["max_e"] for r in results]
    colors   = [LEVEL_COLORS[r["level"]] for r in results]

    x = np.arange(len(labels))
    w = 0.32

    bars1 = ax.bar(x - w / 2, se_vals,  w, label="Semantic Entropy",
                   color=colors, edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + w / 2, max_vals, w, label="Max Possible Entropy",
                   color="#BDC3C7", edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars1, se_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, max_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=9, color="#7F8C8D")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Entropy (nats)", fontsize=9)
    ax.set_title("Entropy Comparison Across Scenarios", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(max_vals) * 1.30)

    # annotation: what high entropy means
    ax.text(0.98, 0.95,
            "High SE  ->  responses cover many\n"
            "           different meanings\n"
            "           ->  possible hallucination",
            transform=ax.transAxes, fontsize=7.5,
            ha="right", va="top", color="#555",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#F0F0F0", edgecolor="#CCC"))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Semantic entropy visualization")
    parser.add_argument("--fast",      action="store_true",
                        help="Use small DeBERTa model (~180 MB)")
    parser.add_argument("--heuristic", action="store_true",
                        help="Use offline content-word heuristic (no downloads)")
    args = parser.parse_args()

    mode = "heuristic" if args.heuristic else ("fast" if args.fast else "large")
    model = load_model(mode)

    print("Running scenarios ...")
    results = []
    for s in SCENARIOS:
        print(f"  {s['label']} ...")
        results.append(run_scenario(s, model))

    # print results to terminal too
    print()
    for s, r in zip(SCENARIOS, results):
        ids = r["sem_ids"]
        print(f"  {s['label']}")
        print(f"    clusters      : {ids}")
        print(f"    distinct      : {len(r['clusters'])}")
        print(f"    semantic ent. : {r['se']:.4f}  (max {r['max_e']:.4f})")
        print(f"    level         : {r['level']}")
        print()

    print("Building visualization ...")

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(17, 16))
    fig.patch.set_facecolor("#F8F9FA")

    fig.suptitle(
        "Semantic Entropy  --  Uncertainty Detection in LLM Responses\n"
        "Farquhar et al., Nature 2024  |  github.com/jlko/semantic_uncertainty",
        fontsize=13, fontweight="bold", y=0.992, color="#2C3E50",
    )

    gs_outer = GridSpec(
        4, 1, figure=fig,
        height_ratios=[1, 1, 1, 0.70],
        hspace=0.60,
        left=0.04, right=0.97, top=0.95, bottom=0.04,
    )

    for row in range(3):
        scenario = SCENARIOS[row]
        result   = results[row]
        lc       = LABEL_COLORS[scenario["label"]]

        gs_row = gs_outer[row].subgridspec(
            1, 3, wspace=0.38,
            width_ratios=[2.8, 2.0, 1.2],
        )
        ax_resp = fig.add_subplot(gs_row[0])
        ax_bars = fig.add_subplot(gs_row[1])
        ax_ent  = fig.add_subplot(gs_row[2])

        ax_resp.set_title(
            f"[{scenario['label']}]   Q: \"{scenario['question']}\"",
            fontsize=9, fontweight="bold", loc="left", color=lc, pad=5,
        )

        draw_responses(ax_resp,      scenario, result)
        draw_cluster_bars(ax_bars,   result)
        draw_gauge(ax_ent,           result)

    ax_summary = fig.add_subplot(gs_outer[3])
    draw_summary(ax_summary, results, SCENARIOS)

    out = "semantic_entropy_visualization.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
    print(f"\nSaved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
