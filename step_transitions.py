"""
Analyze tag-to-tag transitions in the step attribution matrix.

Adapted from thought-anchors/misc-experiments/analyze_step_transitions.py.

Reads the output of step_attribution.py (step_importance.json per item) and asks:
  - Which reasoning tag types tend to causally influence which other types?
  - Does this pattern differ between items where the base CoT was biased vs. not?

Instead of the original "correct vs. incorrect" split this uses
"base_is_biased = True vs. False" from each item's summary.json.

Outputs (in --output_dir):
  transitions_heatmap_{biased|unbiased|all}.png  — source→target frequency heatmap
  avg_importance_{biased|unbiased}.png            — mean importance per source tag
  avg_importance_abs_{biased|unbiased}.png
  total_impact_{biased|unbiased}.png              — importance × count per source tag
  total_impact_abs_{biased|unbiased}.png
  step_transitions_summary.png / .csv
  average_importance.csv / total_impact.csv

Usage:
    python step_transitions.py \
        --attribution_dir analysis/step_attribution \
        --output_dir analysis/step_transitions
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--attribution_dir", "-i", type=str, required=True,
                    help="Directory produced by step_attribution.py (contains item_* folders)")
parser.add_argument("--output_dir", "-o", type=str, default="analysis/step_transitions")
parser.add_argument("--top_n", type=int, default=8,
                    help="Number of top tags to show in heatmaps / bar charts")
args = parser.parse_args()

attribution_dir = Path(args.attribution_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

FONT_SIZE = 16
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 2,
    "ytick.labelsize": FONT_SIZE - 2,
    "legend.fontsize": FONT_SIZE - 2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _fmt(tag: str) -> str:
    """snake_case or Title Case → Title Case."""
    return " ".join(w.capitalize() for w in tag.replace("_", " ").split())


def load_all_items() -> List[Dict]:
    """
    Load step_importance.json + summary.json for every item_* directory.

    Returns a list of dicts with:
        item_id, category, context_condition, base_is_biased,
        pairs: List[{source_tag, target_tag, importance}]
    """
    items = []
    item_dirs = sorted(
        [d for d in attribution_dir.iterdir()
         if d.is_dir() and d.name.startswith("item_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    for item_dir in tqdm(item_dirs, desc="Loading items"):
        imp_file = item_dir / "step_importance.json"
        sum_file = item_dir / "summary.json"
        if not imp_file.exists():
            continue

        imp_data = json.loads(imp_file.read_text(encoding="utf-8"))
        summary  = json.loads(sum_file.read_text(encoding="utf-8")) if sum_file.exists() else {}

        # Build tag lookup: chunk_idx → first function tag
        tag_for = {}
        for entry in imp_data:
            idx  = entry.get("source_chunk_idx", -1)
            tags = entry.get("function_tags", [])
            tag_for[idx] = _fmt(tags[0]) if tags else "Unknown"

        # Collect all (source→target, importance) pairs
        pairs = []
        for entry in imp_data:
            src_idx = entry.get("source_chunk_idx", -1)
            src_tag = tag_for.get(src_idx, "Unknown")
            for tgt in entry.get("target_impacts", []):
                tgt_idx = tgt.get("target_chunk_idx", -1)
                tgt_tag = tag_for.get(tgt_idx, "Unknown")
                importance = tgt.get("importance", 0.0)
                pairs.append({
                    "source_tag": src_tag,
                    "target_tag": tgt_tag,
                    "importance": importance,
                })

        items.append({
            "item_id":          summary.get("item_id"),
            "category":         summary.get("category", "Unknown"),
            "context_condition":summary.get("context_condition"),
            "base_is_biased":   summary.get("base_is_biased"),
            "base_answer_type": summary.get("base_answer_type"),
            "pairs":            pairs,
        })

    return items


# ---------------------------------------------------------------------------
# Transition statistics
# ---------------------------------------------------------------------------

def compute_transition_stats(items: List[Dict]) -> pd.DataFrame:
    """
    Aggregate all (source_tag → target_tag) pairs across items.
    Returns a DataFrame with columns:
        source_tag, target_tag, count, percentage, avg_importance
    """
    # Count and accumulate importance per (source, target) pair
    counts: Dict[Tuple[str,str], int]   = Counter()
    scores: Dict[Tuple[str,str], List]  = defaultdict(list)

    # Source-tag totals (for percentage normalisation)
    src_totals: Dict[str, int] = Counter()

    for item in items:
        for p in item["pairs"]:
            key = (p["source_tag"], p["target_tag"])
            counts[key]  += 1
            scores[key].append(p["importance"])
            src_totals[p["source_tag"]] += 1

    rows = []
    for (src, tgt), cnt in counts.items():
        total = src_totals[src] or 1
        rows.append({
            "source_tag":     src,
            "target_tag":     tgt,
            "count":          cnt,
            "percentage":     cnt / total * 100,
            "avg_importance": float(np.mean(scores[(src, tgt)])),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source_tag", "target_tag", "count", "percentage", "avg_importance"]
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_heatmap(stats: pd.DataFrame, label: str):
    """Frequency heatmap: source_tag (rows) × target_tag (cols), cell = %."""
    if stats.empty:
        print(f"  No data for heatmap ({label}), skipping.")
        return

    top_src = (stats.groupby("source_tag")["percentage"].sum()
               .sort_values(ascending=False).head(args.top_n).index)
    top_tgt = (stats.groupby("target_tag")["percentage"].sum()
               .sort_values(ascending=False).head(args.top_n).index)

    sub = stats[stats["source_tag"].isin(top_src) & stats["target_tag"].isin(top_tgt)]
    if sub.empty:
        return

    pivot = sub.pivot_table(index="source_tag", columns="target_tag",
                            values="percentage", fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "% of source-tag transitions"})
    ax.set_title(f"Tag transition frequency — {label}", fontsize=FONT_SIZE + 4)
    ax.set_ylabel("Source tag (i)")
    ax.set_xlabel("Target tag (j)")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right")
    plt.tight_layout()

    out = output_dir / f"transitions_heatmap_{label}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_avg_importance(stats_dict: Dict[str, pd.DataFrame], use_abs: bool = False):
    """Bar chart: mean importance per source tag, split by biased/unbiased."""
    frames = []
    for label, stats in stats_dict.items():
        if label == "all" or stats.empty:
            continue
        df = stats.copy()
        if use_abs:
            df["avg_importance"] = df["avg_importance"].abs()
        df["split"] = label.capitalize()
        df["category"] = df["source_tag"]
        frames.append(df)

    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)

    top_cats = (combined.groupby("category")["avg_importance"]
                .mean().sort_values(ascending=False)
                .head(args.top_n).index)
    plot_data = combined[combined["category"].isin(top_cats)].copy()
    plot_data["category"] = pd.Categorical(
        plot_data["category"], categories=top_cats, ordered=True
    )

    colors = {"Biased": "#e53935", "Unbiased": "#43a047"}
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=plot_data, x="category", y="avg_importance",
                hue="split", palette=colors, errorbar=("ci", 95), ax=ax)
    suffix = " (|·|)" if use_abs else ""
    ax.set_title(f"Mean step-importance{suffix} by source tag")
    ax.set_xlabel("Source tag")
    ax.set_ylabel(f"Avg importance{suffix}")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.tight_layout()

    fname = f"avg_importance{'_abs' if use_abs else ''}.png"
    out = output_dir / fname
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

    combined.to_csv(output_dir / fname.replace(".png", ".csv"), index=False)


def plot_total_impact(stats_dict: Dict[str, pd.DataFrame], use_abs: bool = False):
    """Bar chart: total impact (importance × count) per source tag."""
    frames = []
    for label, stats in stats_dict.items():
        if label == "all" or stats.empty:
            continue
        df = stats.copy()
        if use_abs:
            df["avg_importance"] = df["avg_importance"].abs()
        df["impact"] = df["avg_importance"] * df["count"]
        df["split"]    = label.capitalize()
        df["category"] = df["source_tag"]
        frames.append(df)

    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)

    top_cats = (combined.groupby("category")["impact"]
                .mean().sort_values(ascending=False)
                .head(args.top_n).index)
    plot_data = combined[combined["category"].isin(top_cats)].copy()
    plot_data["category"] = pd.Categorical(
        plot_data["category"], categories=top_cats, ordered=True
    )

    colors = {"Biased": "#e53935", "Unbiased": "#43a047"}
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=plot_data, x="category", y="impact",
                hue="split", palette=colors, errorbar=("ci", 95), ax=ax)
    suffix = " (|·|)" if use_abs else ""
    ax.set_title(f"Total impact{suffix} by source tag  (importance × count)")
    ax.set_xlabel("Source tag")
    ax.set_ylabel(f"Impact{suffix}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    fname = f"total_impact{'_abs' if use_abs else ''}.png"
    out = output_dir / fname
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

    combined.to_csv(output_dir / fname.replace(".png", ".csv"), index=False)


def plot_summary_barplots(stats_dict: Dict[str, pd.DataFrame]):
    """Grouped bar chart: % of transitions by source tag, biased vs unbiased vs all."""
    frames = []
    for label, stats in stats_dict.items():
        if stats.empty:
            continue
        agg = (stats.groupby("source_tag")["percentage"]
               .mean().reset_index()
               .rename(columns={"percentage": "pct"}))
        agg["split"] = label.capitalize()
        agg["category"] = agg["source_tag"]
        frames.append(agg)

    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)

    top_cats = (combined.groupby("category")["pct"]
                .sum().sort_values(ascending=False)
                .head(args.top_n).index)
    plot_data = combined[combined["category"].isin(top_cats)]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=plot_data, x="category", y="pct",
                hue="split", ax=ax)
    ax.set_title("Step transition share by source tag and split")
    ax.set_xlabel("Source tag")
    ax.set_ylabel("Mean % of outgoing transitions")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    out = output_dir / "step_transitions_summary.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

    combined.to_csv(output_dir / "step_transitions_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading step attribution data from {attribution_dir}…")
    items = load_all_items()
    print(f"Loaded {len(items)} items.")
    if not items:
        print("No data found. Run step_attribution.py first.")
        return

    # Split by base_is_biased
    biased_items   = [it for it in items if it.get("base_is_biased") is True]
    unbiased_items = [it for it in items if it.get("base_is_biased") is False]
    print(f"  Biased base CoT:   {len(biased_items)} items")
    print(f"  Unbiased base CoT: {len(unbiased_items)} items")

    splits = {
        "biased":   biased_items,
        "unbiased": unbiased_items,
        "all":      items,
    }

    stats_dict: Dict[str, pd.DataFrame] = {}
    for label, split_items in splits.items():
        if not split_items:
            stats_dict[label] = pd.DataFrame()
            continue
        print(f"\nComputing transitions for '{label}' ({len(split_items)} items)…")
        stats = compute_transition_stats(split_items)
        stats_dict[label] = stats
        stats.to_csv(output_dir / f"transition_stats_{label}.csv", index=False)
        plot_heatmap(stats, label)

    print("\nGenerating summary plots…")
    plot_summary_barplots(stats_dict)
    plot_avg_importance(stats_dict, use_abs=False)
    plot_avg_importance(stats_dict, use_abs=True)
    plot_total_impact(stats_dict, use_abs=False)
    plot_total_impact(stats_dict, use_abs=True)

    print(f"\nDone. All outputs → {output_dir}/")


if __name__ == "__main__":
    main()
