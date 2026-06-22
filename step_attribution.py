"""
Sentence-to-sentence causal attribution for bias CoT rollouts.

Adapted from thought-anchors/step_attribution.py.

For each pair of chunks (i, j) with i < j, computes:

    importance[i, j] = match_rate(chunk j | chunk i kept)
                     - match_rate(chunk j | chunk i removed)

"Match rate" = fraction of rollouts in which a step semantically similar
to chunk j appears (cosine similarity ≥ threshold).

If chunk i causally drives chunk j, removing chunk i should make j less
likely to appear downstream → positive importance[i, j].

No API calls — works entirely from rollouts already saved by generate_rollouts.py.

Output per item:
    step_attribution/{item_i}/
        step_importance_matrix.png   — N×N heatmap
        outgoing_importance.png      — bar chart: how much each chunk drives future steps
        incoming_importance.png      — bar chart: how much each chunk depends on past steps
        step_importance.json         — full matrix as JSON
        summary.json                 — top influential / top dependent chunks

Usage:
    python step_attribution.py \
        --rollouts_dir bias_rollouts/deepseek-r1/temperature_0.6_top_p_0.95/ambig \
        --output_dir analysis/step_attribution
"""

import multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils import split_solution_into_chunks

# ---------------------------------------------------------------------------
# Visual config
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "Stereotype Activation": "#e53935",
    "Stereotype Rejection": "#43a047",
    "Group Attribution":     "#ff9800",
    "Uncertainty Expression": "#9c27b0",
    "Context Recall":        "#1e88e5",
    "Option Evaluation":     "#00897b",
    "Answer Commitment":     "#795548",
    "Logical Reasoning":     "#c0ca33",
    "Question Interpretation": "#ec407a",
    "Backtracking":          "#607d8b",
}

FONT_SIZE = 16
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 4,
    "axes.labelsize": FONT_SIZE + 2,
    "xtick.labelsize": FONT_SIZE - 2,
    "ytick.labelsize": FONT_SIZE - 2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 14,
    "axes.titlepad": 14,
})
CMAP = "RdBu"
FIGSIZE = (14, 12)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--rollouts_dir", "-i", type=str, required=True,
                    help="Directory produced by generate_rollouts.py (contains item_* folders)")
parser.add_argument("--output_dir", "-o", type=str, default="analysis/step_attribution")
parser.add_argument("--similarity_threshold", "-st", type=float, default=0.7,
                    help="Min cosine similarity to count a step as 'matching' a target chunk")
parser.add_argument("--max_chunks", "-mc", type=int, default=50,
                    help="Max chunks per item to analyse (keeps runtime manageable)")
parser.add_argument("--items", type=str, default=None,
                    help="Comma-separated item indices to process (default: all)")
parser.add_argument("--num_top", type=int, default=None,
                    help="Only plot the N most important chunks per item")
parser.add_argument("--no_cache", action="store_true",
                    help="Recompute even if cached .npz exists")
parser.add_argument("--sentence_model", type=str, default="all-MiniLM-L6-v2")
args = parser.parse_args()

rollouts_dir = Path(args.rollouts_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_item(item_dir: Path) -> Dict:
    f = item_dir / "item.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def load_base(item_dir: Path) -> Dict:
    f = item_dir / "base_solution.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def load_labeled_chunks(item_dir: Path) -> List[Dict]:
    """Load chunks_labeled.json if it exists (produced by analyze_rollouts.py)."""
    f = item_dir / "chunks_labeled.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    # Fallback: bare chunks without labels
    cf = item_dir / "chunks.json"
    if cf.exists():
        data = json.loads(cf.read_text(encoding="utf-8"))
        return [{"chunk_idx": i, "chunk": c, "function_tags": []}
                for i, c in enumerate(data.get("chunks", []))]
    return []


def load_rollouts(item_dir: Path, chunk_idx: int) -> List[Dict]:
    f = item_dir / f"chunk_{chunk_idx}" / "solutions.json"
    if not f.exists():
        return []
    sols = json.loads(f.read_text(encoding="utf-8"))
    return [s for s in sols if "rollout" in s and s["rollout"] and "error" not in s]


# ---------------------------------------------------------------------------
# Core matrix computation
# ---------------------------------------------------------------------------

def compute_importance_matrix(
    item_dir: Path,
    model: SentenceTransformer,
    cache_dir: Path,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Compute N×N step-importance matrix for one item.

    importance[i, j]  =  match_rate(j | i kept) − match_rate(j | i removed)

    Uses .npz cache keyed by similarity threshold.
    """
    cache_file = cache_dir / f"matrix_t{args.similarity_threshold}.npz"
    if not args.no_cache and cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        return data["matrix"], data["chunks"].tolist()

    chunks = load_labeled_chunks(item_dir)
    if not chunks:
        return np.array([]), []

    chunks = chunks[:args.max_chunks]
    N = len(chunks)
    chunk_texts = [c.get("chunk", "") for c in chunks]

    # Pre-embed all original chunks
    print(f"  Embedding {N} chunks…")
    original_embeddings = model.encode(chunk_texts, show_progress_bar=False,
                                       normalize_embeddings=True)

    # Pre-load and embed all rollout steps for every chunk index
    print("  Loading & embedding rollout steps…")
    rollout_data: Dict[int, Dict] = {}
    for idx in range(N):
        rollouts = load_rollouts(item_dir, idx)
        all_steps, rollout_ids, step_ids = [], [], []
        for r_idx, r in enumerate(rollouts):
            steps = split_solution_into_chunks(r["rollout"])
            for s_idx, step in enumerate(steps):
                all_steps.append(step)
                rollout_ids.append(r_idx)
                step_ids.append(s_idx)

        embeddings = (
            model.encode(all_steps, show_progress_bar=False, normalize_embeddings=True)
            if all_steps else np.array([])
        )
        rollout_data[idx] = {
            "steps": all_steps,
            "rollout_ids": rollout_ids,
            "n_rollouts": len(rollouts),
            "embeddings": embeddings,
        }

    # Build matrix
    matrix = np.zeros((N, N))
    print(f"  Computing {N*(N-1)//2} chunk pairs…")

    with tqdm(total=N * (N - 1) // 2, desc="  Pairs", leave=False) as pbar:
        for i in range(N - 1):
            # "kept i"   → resampling started at i+1
            # "removed i" → resampling started at i
            include_data = rollout_data.get(i + 1, {})
            exclude_data = rollout_data.get(i, {})

            inc_emb = include_data.get("embeddings", np.array([]))
            exc_emb = exclude_data.get("embeddings", np.array([]))
            inc_rids = include_data.get("rollout_ids", [])
            exc_rids = exclude_data.get("rollout_ids", [])
            inc_n = include_data.get("n_rollouts", 0)
            exc_n = exclude_data.get("n_rollouts", 0)

            if inc_n == 0 or exc_n == 0 or len(inc_emb) == 0 or len(exc_emb) == 0:
                pbar.update(N - 1 - i)
                continue

            for j in range(i + 1, N):
                target = original_embeddings[j].reshape(1, -1)

                # Similarities for all rollout steps against target chunk j
                inc_sims = cosine_similarity(target, inc_emb)[0]
                exc_sims = cosine_similarity(target, exc_emb)[0]

                # Best similarity per rollout (match = sim ≥ threshold)
                inc_best = [-1.0] * inc_n
                for k, (rid, sim) in enumerate(zip(inc_rids, inc_sims)):
                    if sim >= args.similarity_threshold and sim > inc_best[rid]:
                        inc_best[rid] = float(sim)

                exc_best = [-1.0] * exc_n
                for k, (rid, sim) in enumerate(zip(exc_rids, exc_sims)):
                    if sim >= args.similarity_threshold and sim > exc_best[rid]:
                        exc_best[rid] = float(sim)

                inc_rate = sum(1 for s in inc_best if s >= 0) / inc_n
                exc_rate = sum(1 for s in exc_best if s >= 0) / exc_n

                matrix[i, j] = inc_rate - exc_rate
                pbar.update(1)

    np.savez(cache_file, matrix=matrix, chunks=np.array(chunks, dtype=object))
    return matrix, chunks


# ---------------------------------------------------------------------------
# Importance aggregates
# ---------------------------------------------------------------------------

def outgoing_importance(matrix: np.ndarray) -> np.ndarray:
    """Average importance of chunk i on all future chunks j > i."""
    N = matrix.shape[0]
    out = np.zeros(N)
    for i in range(N - 1):
        future = matrix[i, i + 1:]
        out[i] = np.mean(future) if len(future) > 0 else 0.0
    return out


def incoming_importance(matrix: np.ndarray) -> np.ndarray:
    """Average importance of all past chunks i < j on chunk j."""
    N = matrix.shape[0]
    inc = np.zeros(N)
    for j in range(1, N):
        past = matrix[:j, j]
        inc[j] = np.mean(past) if len(past) > 0 else 0.0
    return inc


def select_top(matrix: np.ndarray, chunks: List[Dict], n: int):
    """Keep only the N most important chunks (by combined importance)."""
    if n is None or n >= len(chunks):
        return matrix, chunks, list(range(len(chunks)))
    combined = outgoing_importance(matrix) + incoming_importance(matrix)
    top_idx = sorted(np.argsort(combined)[-n:])
    return (
        matrix[np.ix_(top_idx, top_idx)],
        [chunks[i] for i in top_idx],
        top_idx,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _tag_color(chunk: Dict) -> str:
    tags = chunk.get("function_tags", [])
    if not tags:
        return "#000000"
    tag = tags[0].replace("_", " ").title()
    return CATEGORY_COLORS.get(tag, "#000000")


def _tag_abbrev(chunk: Dict) -> str:
    tags = chunk.get("function_tags", [])
    if not tags:
        return ""
    return "".join(w[0].upper() for w in tags[0].split("_"))


def plot_matrix(matrix: np.ndarray, chunks: List[Dict], original_indices: List[int],
                out_file: Path, title: str) -> None:
    N = len(chunks)
    plot_m = matrix.T.copy()
    mask = np.triu(np.ones_like(plot_m, dtype=bool))

    labels = [f"{original_indices[i]}-{_tag_abbrev(chunks[i])}" for i in range(N)]
    colors = [_tag_color(c) for c in chunks]

    vmax = max(abs(plot_m.min()), abs(plot_m.max())) or 1.0
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.heatmap(
        plot_m, cmap=CMAP, mask=mask,
        norm=mcolors.Normalize(vmin=-vmax, vmax=vmax),
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "Importance (match rate diff)"},
        ax=ax,
    )
    # White out upper triangle
    for i in range(N):
        for j in range(N):
            if j <= i:
                ax.add_patch(plt.Rectangle((i, j), 1, 1,
                                           fill=True, color="white", zorder=10))
    for tick, col in zip(ax.get_xticklabels(), colors):
        tick.set_color(col)
    for tick, col in zip(ax.get_yticklabels(), colors):
        tick.set_color(col)

    ax.set_xlabel("Source step (i)")
    ax.set_ylabel("Target step (j)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_bar(values: np.ndarray, chunks: List[Dict], original_indices: List[int],
             out_file: Path, title: str, ylabel: str) -> None:
    N = len(values)
    colors = [_tag_color(c) for c in chunks]
    labels = [str(original_indices[i]) for i in range(N)]

    fig, ax = plt.subplots(figsize=(max(10, N * 0.5), 5))
    ax.bar(range(N), values, color=colors, alpha=0.85)
    ax.set_xticks(range(N))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Chunk index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Per-item analysis
# ---------------------------------------------------------------------------

def analyze_item(item_dir: Path, model: SentenceTransformer) -> Optional[Dict]:
    item_idx = item_dir.name.split("_")[1]
    item_out = output_dir / f"item_{item_idx}"
    item_out.mkdir(parents=True, exist_ok=True)
    cache_dir = item_out / "cache"
    cache_dir.mkdir(exist_ok=True)

    print(f"\n[item {item_idx}]")
    matrix, chunks = compute_importance_matrix(item_dir, model, cache_dir)

    if matrix.size == 0 or len(chunks) < 2:
        print(f"  Skipping — insufficient data.")
        return None

    original_indices = list(range(len(chunks)))

    # Optionally keep only top-N chunks
    if args.num_top:
        matrix, chunks, original_indices = select_top(matrix, chunks, args.num_top)

    N = len(chunks)
    out_imp = outgoing_importance(matrix)
    inc_imp = incoming_importance(matrix)
    combined = out_imp + inc_imp

    item = load_item(item_dir)
    base  = load_base(item_dir)
    title_prefix = (
        f"Item {item_idx} | {item.get('category','')} | "
        f"{item.get('context_condition','')} | "
        f"base: {base.get('answer_type','?')}"
    )

    # Plots
    plot_matrix(
        matrix, chunks, original_indices,
        item_out / "step_importance_matrix.png",
        f"{title_prefix}\nSentence→sentence causal importance",
    )
    plot_bar(
        out_imp, chunks, original_indices,
        item_out / "outgoing_importance.png",
        f"{title_prefix}\nOutgoing importance (avg effect on future chunks)",
        "Avg importance on j > i",
    )
    plot_bar(
        inc_imp, chunks, original_indices,
        item_out / "incoming_importance.png",
        f"{title_prefix}\nIncoming importance (avg effect from past chunks)",
        "Avg importance from i < j",
    )

    # JSON export
    importance_data = []
    for i in range(N):
        importance_data.append({
            "source_chunk_idx": original_indices[i],
            "source_chunk": chunks[i].get("chunk", ""),
            "function_tags": chunks[i].get("function_tags", []),
            "outgoing_importance": float(out_imp[i]),
            "incoming_importance": float(inc_imp[i]),
            "combined_importance": float(combined[i]),
            "target_impacts": [
                {"target_chunk_idx": original_indices[j],
                 "importance": float(matrix[i, j])}
                for j in range(N) if j > i
            ],
        })
    (item_out / "step_importance.json").write_text(
        json.dumps(importance_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Summary
    top_out_idx = int(np.argmax(out_imp))
    top_inc_idx = int(np.argmax(inc_imp))
    summary = {
        "item_id": item.get("instance_id"),
        "category": item.get("category"),
        "context_condition": item.get("context_condition"),
        "base_answer_type": base.get("answer_type"),
        "num_chunks_analysed": N,
        "most_influential_chunk": {
            "original_idx": original_indices[top_out_idx],
            "chunk": chunks[top_out_idx].get("chunk", ""),
            "function_tags": chunks[top_out_idx].get("function_tags", []),
            "outgoing_importance": float(out_imp[top_out_idx]),
        },
        "most_dependent_chunk": {
            "original_idx": original_indices[top_inc_idx],
            "chunk": chunks[top_inc_idx].get("chunk", ""),
            "function_tags": chunks[top_inc_idx].get("function_tags", []),
            "incoming_importance": float(inc_imp[top_inc_idx]),
        },
    }
    (item_out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"  Most influential chunk (outgoing): [{original_indices[top_out_idx]}] "
          f"{chunks[top_out_idx].get('chunk','')[:80]}…")
    print(f"  Most dependent chunk (incoming):  [{original_indices[top_inc_idx]}] "
          f"{chunks[top_inc_idx].get('chunk','')[:80]}…")

    del matrix
    gc.collect()
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    item_dirs = sorted(
        [d for d in rollouts_dir.iterdir()
         if d.is_dir() and d.name.startswith("item_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if args.items:
        include = {int(x) for x in args.items.split(",")}
        item_dirs = [d for d in item_dirs if int(d.name.split("_")[1]) in include]

    print(f"Loading embedding model ({args.sentence_model})…")
    model = SentenceTransformer(args.sentence_model)
    model.eval()

    print(f"Processing {len(item_dirs)} items → {output_dir}")
    summaries = []
    for item_dir in tqdm(item_dirs, desc="Items"):
        s = analyze_item(item_dir, model)
        if s:
            summaries.append(s)

    (output_dir / "all_summaries.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDone. {len(summaries)} items analysed → {output_dir}/all_summaries.json")


if __name__ == "__main__":
    main()
