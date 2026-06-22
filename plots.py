"""
Visualizations for bias-thought-anchors experiment on SESGO-small.

Figures:
  1. Base bias rates — grouped bar chart per model × condition
  2. P(pro-stereo) trajectory — line plot across normalized chunk position
  3. Semantic tag importance — bar chart of mean |Δ_RS| by tag
  4. Top individual chunks — scatter of Δ_RS vs Δ_CF coloured by tag
  5. Category × model heatmap of base bias rate
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ANALYSIS_DIR = Path("analysis/sesgo")
FIGURES_DIR  = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

MODELS = {
    "qwen3p7-plus":     "Qwen3.7-Plus",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "claude-sonnet-4-6":"Claude Sonnet 4.6",
}
LANGS = ["es", "en"]
CONDS = ["ambig", "disambig"]
CATS  = ["genero", "clasismo", "racismo", "xenofobia"]

# Okabe-Ito palette (colorblind-safe)
COLORS = {
    "qwen3p7-plus":     "#E69F00",
    "gemini-2.5-flash": "#56B4E9",
    "claude-sonnet-4-6":"#009E73",
    "genero":    "#E69F00",
    "clasismo":  "#56B4E9",
    "racismo":   "#009E73",
    "xenofobia": "#CC79A7",
}
TAG_COLORS = {
    "context_recall":         "#E69F00",
    "logical_reasoning":      "#56B4E9",
    "question_interpretation":"#009E73",
    "option_evaluation":      "#F0E442",
    "uncertainty_expression": "#0072B2",
    "backtracking":           "#D55E00",
    "stereotype_activation":  "#CC79A7",
    "group_attribution":      "#000000",
    "answer_commitment":      "#999999",
    "stereotype_rejection":   "#44AA99",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all():
    rows = []
    for model in MODELS:
        for lang in LANGS:
            for cond in CONDS:
                f = ANALYSIS_DIR / model / lang / cond / "results.json"
                if not f.exists():
                    continue
                for item in json.load(open(f)):
                    n = item["num_chunks"]
                    for ch in item.get("labeled_chunks", []):
                        rows.append({
                            "model":    model,
                            "lang":     lang,
                            "cond":     cond,
                            "category": item.get("category", "unknown"),
                            "item_id":  item["item_id"],
                            "base_is_biased": item.get("base_is_biased", False),
                            "chunk_idx": ch["chunk_idx"],
                            "n_chunks":  n,
                            "pos_norm":  ch["chunk_idx"] / max(n - 1, 1),
                            "p_biased":  ch["p_biased"],
                            "rs": ch["resampling_bias_importance"],
                            "cf": ch["counterfactual_bias_importance"],
                            "kl": ch["counterfactual_bias_kl"],
                            "tags": ch.get("function_tags", ["unknown"]),
                        })
    return rows


# ---------------------------------------------------------------------------
# Figure 1 — Base bias rates grouped bar
# ---------------------------------------------------------------------------

def fig_base_bias_rates(rows):
    bias = defaultdict(lambda: defaultdict(list))
    seen = set()
    for r in rows:
        key = (r["model"], r["lang"], r["cond"], r["item_id"])
        if key in seen:
            continue
        seen.add(key)
        bias[(r["model"], r["lang"])][r["cond"]].append(r["base_is_biased"])

    conditions = ["ES/ambig", "ES/disambig", "EN/ambig", "EN/disambig"]
    model_list = list(MODELS.keys())
    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, model in enumerate(model_list):
        vals = []
        for lang, cond in [("es","ambig"),("es","disambig"),("en","ambig"),("en","disambig")]:
            items = bias[(model, lang)][cond]
            clean = [v for v in items if v is not None]
            vals.append(100 * sum(clean) / len(clean) if clean else 0)
        ax.bar(x + (i - 1) * width, vals, width,
               label=MODELS[model], color=COLORS[model], edgecolor="white")

    ax.set_ylabel("Base answers pro-stereotypical (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 100)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_base_bias_rates.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig1_base_bias_rates.png", dpi=150, bbox_inches="tight")
    print("Saved fig1_base_bias_rates")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — P(pro-stereo) trajectory across normalised chunk position
# ---------------------------------------------------------------------------

def fig_trajectory(rows):
    BIN_N = 10
    bins = np.linspace(0, 1, BIN_N + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, model in zip(axes, MODELS):
        for cat in CATS:
            vals = [r for r in rows
                    if r["model"] == model and r["cond"] == "disambig"
                    and r["category"] == cat]
            if not vals:
                continue
            sums = np.zeros(BIN_N)
            cnts = np.zeros(BIN_N)
            for r in vals:
                b = min(int(r["pos_norm"] * BIN_N), BIN_N - 1)
                sums[b] += r["p_biased"]
                cnts[b] += 1
            means = np.where(cnts > 0, sums / cnts, np.nan)
            ax.plot(bin_centers, means, marker="o", markersize=4,
                    label=cat, color=COLORS[cat], linewidth=1.8)

        ax.set_title(MODELS[model], fontsize=10)
        ax.set_xlabel("Normalised chunk position")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("P(pro-stereo)")
    handles = [mpatches.Patch(color=COLORS[c], label=c) for c in CATS]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08), framealpha=0.9)
    fig.suptitle("P(pro-stereo) along CoT — disambig condition", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_trajectory.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig2_trajectory.png", dpi=150, bbox_inches="tight")
    print("Saved fig2_trajectory")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Mean |Δ_RS| by semantic tag
# ---------------------------------------------------------------------------

def fig_tag_importance(rows):
    tag_rs = defaultdict(list)
    for r in rows:
        for tag in r["tags"]:
            tag_rs[tag].append(abs(r["rs"]))

    tags_filtered = {t: v for t, v in tag_rs.items() if len(v) >= 5}
    tags_sorted = sorted(tags_filtered, key=lambda t: np.mean(tags_filtered[t]), reverse=True)

    means  = [np.mean(tags_filtered[t]) for t in tags_sorted]
    sems   = [np.std(tags_filtered[t]) / np.sqrt(len(tags_filtered[t])) for t in tags_sorted]
    colors = [TAG_COLORS.get(t, "#aaaaaa") for t in tags_sorted]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(tags_sorted))
    ax.bar(x, means, yerr=sems, capsize=4, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tags_sorted], fontsize=8)
    ax.set_ylabel("Mean |Δ_RS| (resampling bias importance)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_tag_importance.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig3_tag_importance.png", dpi=150, bbox_inches="tight")
    print("Saved fig3_tag_importance")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Scatter Δ_RS vs Δ_CF coloured by primary tag
# ---------------------------------------------------------------------------

def fig_scatter(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, model in zip(axes, MODELS):
        vals = [r for r in rows if r["model"] == model and r["cond"] == "disambig"]
        if not vals:
            ax.set_title(MODELS[model], fontsize=10)
            continue
        xs = [r["rs"] for r in vals]
        ys = [r["cf"] for r in vals]
        cs = [TAG_COLORS.get(r["tags"][0] if r["tags"] else "unknown", "#aaaaaa")
              for r in vals]
        ax.scatter(xs, ys, c=cs, alpha=0.6, s=25, edgecolors="none")
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.axvline(0, color="gray", linewidth=0.6)
        ax.set_title(MODELS[model], fontsize=10)
        ax.set_xlabel("Δ_RS")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Δ_CF")
    shown_tags = {r["tags"][0] for r in rows if r["tags"]}
    handles = [mpatches.Patch(color=TAG_COLORS.get(t, "#aaa"),
                               label=t.replace("_", " "))
               for t in sorted(shown_tags) if t in TAG_COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.12), fontsize=8, framealpha=0.9)
    fig.suptitle("Δ_RS vs Δ_CF per chunk — disambig condition", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_scatter.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig4_scatter.png", dpi=150, bbox_inches="tight")
    print("Saved fig4_scatter")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Category × model heatmap of base bias rate (disambig)
# ---------------------------------------------------------------------------

def fig_heatmap_categories(rows):
    bias = defaultdict(lambda: defaultdict(list))
    seen = set()
    for r in rows:
        key = (r["model"], r["category"], r["cond"], r["item_id"])
        if key in seen:
            continue
        seen.add(key)
        if r["cond"] == "disambig":
            bias[r["model"]][r["category"]].append(r["base_is_biased"])

    model_list = list(MODELS.keys())
    data = np.full((len(CATS), len(model_list)), np.nan)
    for j, model in enumerate(model_list):
        for i, cat in enumerate(CATS):
            items = bias[model][cat]
            clean = [v for v in items if v is not None]
            if clean:
                data[i, j] = 100 * sum(clean) / len(clean)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, vmin=0, vmax=100, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(model_list)))
    ax.set_xticklabels([MODELS[m] for m in model_list], rotation=15, ha="right", fontsize=9)
    ax.set_yticks(range(len(CATS)))
    ax.set_yticklabels(CATS)
    for i in range(len(CATS)):
        for j in range(len(model_list)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=9, color="white" if val > 60 else "black")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
    plt.colorbar(im, ax=ax, label="% pro-stereo (disambig, ES+EN combined)")
    ax.set_title("Base bias rate by category × model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_heatmap_categories.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig5_heatmap_categories.png", dpi=150, bbox_inches="tight")
    print("Saved fig5_heatmap_categories")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data…")
    rows = load_all()
    print(f"Loaded {len(rows)} chunk-level rows")

    fig_base_bias_rates(rows)
    fig_trajectory(rows)
    fig_tag_importance(rows)
    fig_scatter(rows)
    fig_heatmap_categories(rows)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
