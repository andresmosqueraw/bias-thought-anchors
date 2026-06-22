"""
Statistical analysis and position-scatter plots for bias thought-anchor results.

Adapted from thought-anchors/sentence_scatter_and_ttests.py.

Two analyses:
1. Paired t-tests: for each pair of function tags, is one significantly more
   important than the other? Paired across items (controls for item difficulty).
2. Position scatter: mean importance vs. mean normalized position per tag,
   with SEM error bars — reveals *where* in the CoT each reasoning type tends
   to appear and how influential it is.

Usage:
    python stats_and_scatter.py \
        --results_dir analysis/ambig \
        --output_dir figures/stats \
        --metric counterfactual_bias_importance
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", "-i", type=str, required=True,
                    help="Directory containing results.json from analyze_rollouts.py")
parser.add_argument("--output_dir", "-o", type=str, default="figures/stats")
parser.add_argument(
    "--metric", type=str,
    default="counterfactual_bias_importance",
    choices=[
        "counterfactual_bias_importance",
        "resampling_bias_importance",
        "counterfactual_bias_kl",
    ],
)
parser.add_argument(
    "--convergence_threshold", type=float, default=0.85,
    help="Items whose p_biased stays above this for --convergence_n consecutive "
         "chunks are flagged as 'converged' and excluded from the analysis "
         "(analogous to thought-anchors' convergence filter for math accuracy).",
)
parser.add_argument("--convergence_n", type=int, default=3)
args = parser.parse_args()

results_dir = Path(args.results_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

FONT_SIZE = 18
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 4,
    "axes.labelsize": FONT_SIZE + 2,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 16,
    "axes.titlepad": 16,
})

CATEGORY_COLORS = {
    "Stereotype Activation":  "#e53935",
    "Stereotype Rejection":   "#43a047",
    "Group Attribution":      "#ff9800",
    "Uncertainty Expression": "#9c27b0",
    "Context Recall":         "#1e88e5",
    "Option Evaluation":      "#00897b",
    "Answer Commitment":      "#795548",
    "Logical Reasoning":      "#c0ca33",
    "Question Interpretation":"#ec407a",
    "Backtracking":           "#607d8b",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results():
    f = results_dir / "results.json"
    if not f.exists():
        raise FileNotFoundError(f"results.json not found in {results_dir}")
    return json.loads(f.read_text(encoding="utf-8"))


def is_converged_flag(p_biased_sequence: list, threshold: float, n: int) -> list:
    """
    Return a per-chunk boolean: True once the model has been in an extreme
    p_biased state (> threshold OR < 1-threshold) for `n` consecutive chunks.
    Mirrors thought-anchors' convergence concept (high/low accuracy for 5+ steps).
    """
    flags = []
    streak = 0
    committed = False
    for p in p_biased_sequence:
        if p > threshold or p < (1 - threshold):
            streak += 1
        else:
            streak = 0
        if streak >= n:
            committed = True
        flags.append(committed)
    return flags


def build_dataframe(results: list) -> pd.DataFrame:
    rows = []
    for r in tqdm(results, desc="Building DataFrame"):
        chunks = r.get("labeled_chunks", [])
        total = len(chunks)
        if total < 2:
            continue

        p_biased_seq = [c.get("p_biased", 0.0) for c in chunks]
        converged_flags = is_converged_flag(
            p_biased_seq, args.convergence_threshold, args.convergence_n
        )

        prev_p = None
        for chunk_idx, (chunk, converged) in enumerate(zip(chunks, converged_flags)):
            tags = chunk.get("function_tags", [])
            tag = (tags[0] if tags else "unknown").replace("_", " ").title()

            p = chunk.get("p_biased", 0.0)
            delta_p = (p - prev_p) if prev_p is not None else None
            prev_p = p

            rows.append({
                "item_id":    r.get("item_id"),
                "category":   r.get("category", "Unknown"),
                "context_condition": r.get("context_condition"),
                "base_is_biased": r.get("base_is_biased", False),
                "chunk_idx":  chunk_idx,
                "function_tag": tag,
                "p_biased":   p,
                "delta_p_biased": delta_p,
                "is_converged": converged,
                "normalized_position": chunk_idx / (total - 1),
                "counterfactual_bias_importance":
                    chunk.get("counterfactual_bias_importance", 0.0),
                "resampling_bias_importance":
                    chunk.get("resampling_bias_importance", 0.0),
                "counterfactual_bias_kl":
                    chunk.get("counterfactual_bias_kl", 0.0),
                "chunk_length": len(chunk.get("chunk", "")),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 1: Paired t-tests across function tags
# ---------------------------------------------------------------------------

SKIP_TAGS = {"Answer Commitment", "Question Interpretation", "Unknown"}

def paired_ttests(df: pd.DataFrame, key: str):
    """
    Paired t-test for each pair of function tags: is tag A significantly
    more important than tag B, controlling for item identity?

    Mirrors thought-anchors/sentence_scatter_and_ttests.py::paired_ttests().
    Uses the median importance per (item, tag) pair, then aligns by item.
    """
    df_active = df[~df["is_converged"]].copy()
    df_active[key] = df_active[key].abs()

    df_grp = (
        df_active.groupby(["item_id", "function_tag"])[key]
        .median()
        .reset_index()
    )

    tags = [t for t in df_grp["function_tag"].unique() if t not in SKIP_TAGS]
    results = []

    for i, tag0 in enumerate(tags):
        for tag1 in tags[i + 1:]:
            df0 = df_grp[df_grp["function_tag"] == tag0].set_index("item_id")
            df1 = df_grp[df_grp["function_tag"] == tag1].set_index("item_id")
            common = df0.index.intersection(df1.index)
            if len(common) < 5:
                continue
            t_stat, p_val = stats.ttest_rel(
                df0.loc[common, key], df1.loc[common, key], nan_policy="omit"
            )
            M0 = df0.loc[common, key].mean()
            M1 = df1.loc[common, key].mean()
            results.append({
                "tag_A": tag0, "M_A": M0,
                "tag_B": tag1, "M_B": M1,
                "t": t_stat, "p": p_val, "n_items": len(common),
            })

    if not results:
        print("Not enough overlapping items for t-tests.")
        return

    df_res = pd.DataFrame(results).sort_values("p")

    print(f"\n{'='*70}")
    print(f"Paired t-tests: {key} (abs value), excluding converged chunks")
    print(f"{'='*70}")
    for _, row in df_res.iterrows():
        sig = "***" if row["p"] < 0.001 else ("**" if row["p"] < 0.01
              else ("*" if row["p"] < 0.05 else ""))
        print(
            f"  {row['tag_A']} ({row['M_A']:.4f}) vs "
            f"{row['tag_B']} ({row['M_B']:.4f}): "
            f"t={row['t']:.2f}, p={row['p']:.4f} {sig}  [n={row['n_items']}]"
        )

    df_res.to_csv(output_dir / f"ttests_{key}.csv", index=False)
    print(f"\nSaved → {output_dir}/ttests_{key}.csv")


# ---------------------------------------------------------------------------
# Analysis 2: Position-importance scatter (one point per tag, with SEM bars)
# ---------------------------------------------------------------------------

PLOT_TAGS = {
    "Stereotype Activation",
    "Stereotype Rejection",
    "Group Attribution",
    "Uncertainty Expression",
    "Context Recall",
    "Logical Reasoning",
}

def plot_scatter(df: pd.DataFrame, key: str):
    """
    Scatter plot: mean importance vs. mean normalized position per function tag.
    Error bars = SEM across items.

    Mirrors thought-anchors/sentence_scatter_and_ttests.py::plot_scatter().
    """
    df_active = df[~df["is_converged"]].copy()
    df_active[key] = df_active[key].abs()
    df_active = df_active[df_active["function_tag"].isin(PLOT_TAGS)]

    # Median per (item, tag), then mean/SEM across items
    df_grp = (
        df_active.groupby(["item_id", "function_tag"])[[key, "normalized_position"]]
        .median()
        .reset_index()
    )
    df_M = df_grp.groupby("function_tag")[[key, "normalized_position"]].mean()
    df_SE = df_grp.groupby("function_tag")[[key, "normalized_position"]].sem()

    fig, ax = plt.subplots(figsize=(9, 7))

    for tag in PLOT_TAGS:
        if tag not in df_M.index:
            continue
        color = CATEGORY_COLORS.get(tag, "#666666")
        ax.errorbar(
            df_M.loc[tag, "normalized_position"],
            df_M.loc[tag, key],
            xerr=df_SE.loc[tag, "normalized_position"],
            yerr=df_SE.loc[tag, key],
            fmt="o", markersize=10, alpha=0.8, capsize=5,
            label=tag, color=color, linewidth=1.5,
        )

    # Auto-zoom axes around the data
    xs = df_M["normalized_position"]
    xe = df_SE["normalized_position"]
    ys = df_M[key]
    ye = df_SE[key]
    x_range = (xs - xe).min(), (xs + xe).max()
    y_range = (ys - ye).min(), (ys + ye).max()
    x_pad = (x_range[1] - x_range[0]) * 0.15 or 0.05
    y_pad = (y_range[1] - y_range[0]) * 0.15 or 0.01
    ax.set_xlim(x_range[0] - x_pad, x_range[1] + x_pad)
    ax.set_ylim(y_range[0] - y_pad, y_range[1] + y_pad)

    metric_label = key.replace("_", " ").replace("bias ", "").title()
    ax.set_xlabel("Normalised position in CoT (0 = start, 1 = end)")
    ax.set_ylabel(f"|{metric_label}|")
    ax.set_title("Reasoning tag: importance vs. position")
    ax.legend(frameon=False, loc="best", fontsize=FONT_SIZE - 4)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = output_dir / f"scatter_position_{key}.pdf"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Bonus: p_biased trajectory across converged vs non-converged items
# ---------------------------------------------------------------------------

def plot_convergence_breakdown(df: pd.DataFrame):
    """Bar chart: fraction of chunks flagged as 'converged' per category."""
    agg = (
        df.groupby("category")["is_converged"]
        .mean()
        .sort_values(ascending=False)
    )
    n_per_cat = df.groupby("category")["item_id"].nunique()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(agg.index, agg.values * 100, color="steelblue", alpha=0.8)
    for bar, cat in zip(bars, agg.index):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"n={n_per_cat[cat]}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("% chunks in converged state")
    ax.set_xlabel("EsBBQ Category")
    ax.set_title(f"Fraction of CoT spent 'converged'\n"
                 f"(|p_biased| > {args.convergence_threshold} for "
                 f"{args.convergence_n}+ consecutive chunks)")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    out = output_dir / "convergence_by_category.pdf"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = load_results()
    print(f"Loaded {len(results)} items from {results_dir}/results.json")

    df = build_dataframe(results)
    print(f"DataFrame: {len(df)} chunk rows, "
          f"{df['item_id'].nunique()} items, "
          f"{df['function_tag'].nunique()} unique tags")
    print(f"Tags: {sorted(df['function_tag'].unique())}")
    print(f"Converged chunks: {df['is_converged'].mean()*100:.1f}%")

    key = args.metric
    paired_ttests(df, key)
    plot_scatter(df, key)
    plot_convergence_breakdown(df)

    # Save the full DataFrame for further analysis
    df.to_csv(output_dir / "chunks_df.csv", index=False)
    print(f"\nFull chunk DataFrame → {output_dir}/chunks_df.csv")


if __name__ == "__main__":
    main()
