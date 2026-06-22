"""
Analyze CoT rollouts for bias thought-anchor identification.

Adapted from thought-anchors/analyze_rollouts.py.
Key differences:
  - Importance metric: Δ P(pro-stereo) instead of Δ accuracy.
  - KL divergence computed over 3-class distribution
    {pro-stereo, anti-stereo, unknown} instead of answer strings.
  - Chunk labeling uses CHUNK_LABEL_PROMPT (social-reasoning tags).
  - Groups results by EsBBQ category (Gender, Age, etc.).

Usage:
    python analyze_rollouts.py \
        --rollouts_dir bias_rollouts/deepseek-r1/temperature_0.6_top_p_0.95/ambig \
        --output_dir analysis/ambig
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from data_utils import answer_type_distribution, p_biased
from prompts import CHUNK_LABEL_PROMPT

load_dotenv()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--rollouts_dir", "-i", type=str, required=True,
                    help="Directory produced by generate_rollouts.py")
parser.add_argument("--output_dir", "-o", type=str, default="analysis",
                    help="Where to save analysis results")
parser.add_argument("--similarity_threshold", type=float, default=0.8,
                    help="Cosine similarity below which two chunks are 'different'")
parser.add_argument("--sentence_model", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--force_relabel", action="store_true",
                    help="Re-label chunks even if chunks_labeled.json exists")
parser.add_argument("--use_existing_metrics", action="store_true",
                    help="Skip metric recalculation if chunks_labeled.json exists")
parser.add_argument("--items", type=str, default=None,
                    help="Comma-separated item indices to analyse (default: all)")
parser.add_argument("--claude_model", type=str, default="claude-haiku-4-5-20251001",
                    help="Claude model for chunk labeling (haiku for cost, sonnet for quality)")
args = parser.parse_args()

rollouts_dir = Path(args.rollouts_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_embed_model: Optional[SentenceTransformer] = None
ANSWER_TYPES = ["pro-stereo", "anti-stereo", "unknown"]
ALPHA = 1e-9

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(args.sentence_model)
    return _embed_model


def embed_texts(texts: List[str]) -> np.ndarray:
    return get_embed_model().encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # already L2-normalized


# ---------------------------------------------------------------------------
# KL divergence over 3-class bias distribution
# ---------------------------------------------------------------------------

def kl_divergence_bias(sols1: List[Dict], sols2: List[Dict]) -> float:
    """KL(P || Q) where P/Q are distributions over {pro-stereo, anti-stereo, unknown}."""
    def counts(sols):
        c = {t: 0 for t in ANSWER_TYPES}
        for s in sols:
            t = s.get("answer_type")
            if t in c:
                c[t] += 1
        return c

    c1, c2 = counts(sols1), counts(sols2)
    n1, n2 = sum(c1.values()), sum(c2.values())
    if n1 == 0 or n2 == 0:
        return 0.0

    V = len(ANSWER_TYPES)
    kl = 0.0
    for t in ANSWER_TYPES:
        p = (c1[t] + ALPHA) / (n1 + ALPHA * V)
        q = (c2[t] + ALPHA) / (n2 + ALPHA * V)
        p_raw = c1[t] / n1
        if p_raw > 0:
            kl += p_raw * math.log(p / q)
    return max(0.0, kl)


# ---------------------------------------------------------------------------
# Importance metrics
# ---------------------------------------------------------------------------

def resampling_bias_importance(chunk_idx: int, chunk_p_biased: Dict[int, float]) -> float:
    """Δ P(pro-stereo) between this chunk and the next one."""
    next_chunks = [i for i in chunk_p_biased if i > chunk_idx]
    if not next_chunks or chunk_idx not in chunk_p_biased:
        return 0.0
    return chunk_p_biased[min(next_chunks)] - chunk_p_biased[chunk_idx]


def counterfactual_bias_importance(
    chunk_idx: int,
    chunk_info: Dict[int, List[Dict]],
    embed_cache: Dict[str, np.ndarray],
    chunk_p_biased: Dict[int, float],
) -> Tuple[float, float, float]:
    """
    Compare rollouts where the resampled chunk is semantically *different*
    from the removed chunk (dissimilar) vs. the next chunk's distribution.

    Returns:
        (importance, different_fraction, overdeterminedness)
    """
    if chunk_idx not in chunk_info:
        return 0.0, 0.0, 0.0

    next_chunks = [i for i in chunk_info if i > chunk_idx]
    if not next_chunks:
        return 0.0, 0.0, 0.0
    next_idx = min(next_chunks)

    dissimilar_sols, resampled_texts = [], []
    for sol in chunk_info[chunk_idx]:
        removed = sol.get("chunk_removed", "")
        resampled = sol.get("chunk_resampled", "")
        if removed not in embed_cache or resampled not in embed_cache:
            continue
        sim = cosine_sim(embed_cache[removed], embed_cache[resampled])
        resampled_texts.append(resampled)
        if sim < args.similarity_threshold:
            dissimilar_sols.append(sol)

    n = len(resampled_texts)
    different_fraction = len(dissimilar_sols) / n if n > 0 else 0.0
    overdeterminedness = 1.0 - len(set(resampled_texts)) / n if n > 0 else 0.0

    next_sols = chunk_info.get(next_idx, [])
    if not dissimilar_sols or not next_sols:
        return 0.0, different_fraction, overdeterminedness

    importance = p_biased(dissimilar_sols) - p_biased(next_sols)
    return importance, different_fraction, overdeterminedness


def counterfactual_kl_importance(
    chunk_idx: int,
    chunk_info: Dict[int, List[Dict]],
    embed_cache: Dict[str, np.ndarray],
) -> float:
    """KL variant of counterfactual importance (3-class bias distribution)."""
    if chunk_idx not in chunk_info:
        return 0.0
    next_chunks = [i for i in chunk_info if i > chunk_idx]
    if not next_chunks:
        return 0.0
    next_idx = min(next_chunks)

    dissimilar_sols, similar_sols = [], []
    for sol in chunk_info[chunk_idx]:
        removed = sol.get("chunk_removed", "")
        resampled = sol.get("chunk_resampled", "")
        if removed not in embed_cache or resampled not in embed_cache:
            continue
        sim = cosine_sim(embed_cache[removed], embed_cache[resampled])
        (dissimilar_sols if sim < args.similarity_threshold else similar_sols).append(sol)

    next_sols = chunk_info.get(next_idx, [])
    if not dissimilar_sols or not next_sols:
        return 0.0
    return kl_divergence_bias(dissimilar_sols, next_sols + similar_sols)


# ---------------------------------------------------------------------------
# Chunk labeling (Claude)
# Assigns interpretability tags to each reasoning step, e.g.:
#   stereotype_activation, group_attribution, uncertainty_expression, backtracking…
# This is optional — the importance metrics above work without it.
# It enables the violin plots and heatmaps grouped by reasoning type in plots.py.
# ---------------------------------------------------------------------------

def label_chunk(item: Dict, chunks: List[str], chunk_idx: int) -> Dict:
    """Call Claude to assign function tags to a single chunk."""
    full_chunked_text = "".join(
        f"Chunk {i}:\n{c}\n\n" for i, c in enumerate(chunks)
    )
    prompt = CHUNK_LABEL_PROMPT.format(
        question=item.get("question", ""),
        ans0=item.get("ans0", ""),
        ans1=item.get("ans1", ""),
        ans2=item.get("ans2", ""),
        full_chunked_text=full_chunked_text,
        chunk_idx=chunk_idx,
    )
    try:
        resp = client.messages.create(
            model=args.claude_model,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        key = str(chunk_idx)
        if key in result:
            return result[key]
        if "function_tags" in result:
            return result
        return {"function_tags": ["unknown"], "depends_on": []}
    except Exception as e:
        print(f"Label error for chunk {chunk_idx}: {e}")
        return {"function_tags": ["unknown"], "depends_on": []}


# ---------------------------------------------------------------------------
# Per-item analysis
# ---------------------------------------------------------------------------

def analyze_item(item_dir: Path) -> Optional[Dict]:
    item_file   = item_dir / "item.json"
    base_file   = item_dir / "base_solution.json"
    chunks_file = item_dir / "chunks.json"

    if not (item_file.exists() and base_file.exists() and chunks_file.exists()):
        print(f"[{item_dir.name}] Missing files, skipping.")
        return None

    with open(item_file,   encoding="utf-8") as f: item   = json.load(f)
    with open(base_file,   encoding="utf-8") as f: base   = json.load(f)
    with open(chunks_file, encoding="utf-8") as f: chunks = json.load(f).get("chunks", [])

    if not chunks:
        return None

    chunk_dirs = {
        int(d.name.split("_")[1]): d
        for d in item_dir.iterdir()
        if d.is_dir() and d.name.startswith("chunk_")
    }
    valid_indices = sorted(chunk_dirs.keys())
    if len(valid_indices) < 2:
        print(f"[{item_dir.name}] Too few chunk dirs, skipping.")
        return None

    # Load rollout solutions
    chunk_info: Dict[int, List[Dict]] = {}
    chunk_p_biased_map: Dict[int, float] = {}
    for idx in valid_indices:
        sols_file = chunk_dirs[idx] / "solutions.json"
        if not sols_file.exists():
            continue
        with open(sols_file, encoding="utf-8") as f:
            solutions = json.load(f)
        valid = [s for s in solutions if "error" not in s and s.get("answer") is not None]
        if valid:
            chunk_info[idx] = valid
            chunk_p_biased_map[idx] = p_biased(valid)

    if not chunk_info:
        return None

    # Build embedding cache
    texts_to_embed = {
        val
        for sols in chunk_info.values()
        for sol in sols
        for key in ("chunk_removed", "chunk_resampled")
        if isinstance(val := sol.get(key, ""), str) and val
    }
    texts_list = list(texts_to_embed)
    embed_cache = dict(zip(texts_list, embed_texts(texts_list)))

    # Load or initialize labeled chunks
    labeled_file = item_dir / "chunks_labeled.json"
    if labeled_file.exists() and not args.force_relabel:
        with open(labeled_file, encoding="utf-8") as f:
            labeled_chunks = json.load(f)
        if args.use_existing_metrics:
            print(f"[{item_dir.name}] Using existing metrics.")
            return _build_result(item_dir, item, base, chunks, labeled_chunks)
    else:
        labeled_chunks = []

    # Compute metrics and labels per chunk
    print(f"[{item_dir.name}] Computing metrics for {len(valid_indices)} chunks…")
    new_labeled = []
    for chunk_idx in valid_indices:
        chunk_text = chunks[chunk_idx] if chunk_idx < len(chunks) else ""

        # Reuse existing labels if available
        existing = next((c for c in labeled_chunks if c.get("chunk_idx") == chunk_idx), None)
        if existing and not args.force_relabel:
            tags      = existing.get("function_tags", ["unknown"])
            depends_on = existing.get("depends_on", [])
        else:
            label_result = label_chunk(item, chunks, chunk_idx)
            tags       = label_result.get("function_tags", ["unknown"])
            depends_on = label_result.get("depends_on", [])

        rs = resampling_bias_importance(chunk_idx, chunk_p_biased_map)
        cf, diff_frac, overdetermined = counterfactual_bias_importance(
            chunk_idx, chunk_info, embed_cache, chunk_p_biased_map
        )
        cf_kl = counterfactual_kl_importance(chunk_idx, chunk_info, embed_cache)
        dist  = answer_type_distribution(chunk_info.get(chunk_idx, []))

        new_labeled.append({
            "chunk_idx": chunk_idx,
            "chunk": chunk_text,
            "function_tags": tags,
            "depends_on": depends_on,
            "p_biased": chunk_p_biased_map.get(chunk_idx, 0.0),
            "answer_type_distribution": dist,
            "resampling_bias_importance": rs,
            "counterfactual_bias_importance": cf,
            "counterfactual_bias_kl": cf_kl,
            "different_trajectories_fraction": diff_frac,
            "overdeterminedness": overdetermined,
        })

    with open(labeled_file, "w", encoding="utf-8") as f:
        json.dump(new_labeled, f, indent=2, ensure_ascii=False)

    return _build_result(item_dir, item, base, chunks, new_labeled)


def _build_result(item_dir, item, base, chunks, labeled_chunks) -> Dict:
    return {
        "item_id": item.get("instance_id") or item.get("item_id") or item_dir.name,
        "category": item.get("category") or item.get("bias_type"),
        "context_condition": item.get("context_condition"),
        "question_polarity": item.get("question_polarity"),
        "base_answer_type": base.get("answer_type"),
        "base_is_biased": base.get("is_biased"),
        "num_chunks": len(chunks),
        "labeled_chunks": labeled_chunks,
    }


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

    print(f"Analysing {len(item_dirs)} items from {rollouts_dir}…")
    results = []
    for item_dir in tqdm(item_dirs, desc="Analysing"):
        result = analyze_item(item_dir)
        if result:
            results.append(result)

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} item results → {output_dir / 'results.json'}")

    by_cat: Dict[str, List] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    print("\n--- Summary by category ---")
    for cat, rs in sorted(by_cat.items()):
        n_biased = sum(1 for r in rs if r.get("base_is_biased"))
        print(f"  {cat}: {len(rs)} items, "
              f"{n_biased} biased base answers ({100*n_biased/len(rs):.0f}%)")

    summary = {
        cat: {
            "n_items": len(rs),
            "n_biased_base": sum(1 for r in rs if r.get("base_is_biased")),
            "pct_biased_base": sum(1 for r in rs if r.get("base_is_biased")) / len(rs),
        }
        for cat, rs in by_cat.items()
    }
    with open(output_dir / "category_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
