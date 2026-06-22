"""
Convert bias-thought-anchors (SESGO) experiment outputs into the JSON format
consumed by the thought-anchors.com Next.js visualizer.

For each item we emit, under
  <WEB>/src/app/data/<model>/<lang>_<cond>/<item_id>/ :
    scenario.json        - bias scenario metadata (context/question/options)
    base_solution.json   - prompt + base chain-of-thought
    chunks_labeled.json   - per-chunk tags + importance (aliased to site fields)
    summary.json          - num_chunks + top influential steps

Mapping note: the site reads each chunk's display importance from
`counterfactual_importance_category_kl`. We map that to |Δ_RS|
(absolute resampling bias importance), the paper's headline per-chunk metric,
so the most bias-relevant steps are highlighted. Original metrics are kept
verbatim alongside the aliases.

Two site features are left empty because the experiment did not store the
underlying data: the pairwise step->step attribution graph
(step_importance.json) and the resampled-continuation viewer
(chunks_resampled.json). Their absence is handled gracefully by the site.
"""

import json
from pathlib import Path

EXP = Path("/home/andrew/Documents/docs/1-super-related-work/papers-manual/"
           "ai-safety-research/experiments/bias-thought-anchors")
WEB = Path("/home/andrew/Documents/docs/1-super-related-work/papers-manual/"
           "ai-safety-research/experiments/bias-thought-anchors.com/src/app/data")

TEMP = "temperature_0.6_top_p_0.95"
MODELS = ["qwen3p7-plus", "gemini-2.5-flash", "claude-sonnet-4-6"]
LANGS = ["es", "en"]
CONDS = ["ambig", "disambig"]

# The "Solution" toggle groups items by the answer the model gave
# (analogous to Correct/Incorrect in the math demo).
BUCKET_MAP = {
    "pro-stereo": "pro_stereo",
    "anti-stereo": "anti_stereo",
    "unknown": "unknown",
    "unparseable": "unparseable",
}


def pretty_tag(tags):
    return tags[0].replace("_", " ") if tags else "step"


def convert():
    n_items = 0
    for model in MODELS:
        counters = {}  # per answer-type bucket, for re-numbering items
        for lang in LANGS:
            for cond in CONDS:
                ana = EXP / f"analysis/sesgo/{model}/{lang}/{cond}/results.json"
                if not ana.exists():
                    continue
                items = json.load(open(ana, encoding="utf-8"))
                for it in items:
                    iid = it["item_id"]
                    soltype = BUCKET_MAP.get(it.get("base_answer_type"),
                                             "unknown")
                    item_key = f"item_{counters.get(soltype, 0)}"
                    counters[soltype] = counters.get(soltype, 0) + 1
                    roll = EXP / ("bias_rollouts/sesgo/"
                                  f"{model}/{TEMP}/{lang}/{cond}/{iid}")
                    if not roll.exists():
                        print(f"  skip (no rollout): {model}/{lang}/{cond}/{iid}")
                        continue
                    meta = json.load(open(roll / "item.json", encoding="utf-8"))
                    base = json.load(open(roll / "base_solution.json",
                                          encoding="utf-8"))
                    chunks = it["labeled_chunks"]

                    # Reconstruct the model's answer from the authoritative
                    # base_answer_type + polarity + target/other indices, so it
                    # is always consistent with the Solution bucket. (Parsing the
                    # letter from text is unreliable.)
                    bat = it.get("base_answer_type")
                    pol = it.get("question_polarity")
                    tgt, oth = meta.get("target"), meta.get("other")
                    if bat == "unknown":
                        chosen = 2          # option C: "not enough information"
                    elif bat == "pro-stereo":
                        chosen = tgt if pol == "neg" else oth
                    elif bat == "anti-stereo":
                        chosen = oth if pol == "neg" else tgt
                    else:
                        chosen = None       # unparseable
                    opt_list = [meta.get("ans0"), meta.get("ans1"),
                                meta.get("ans2")]
                    if chosen is not None and 0 <= chosen < 3:
                        letter = "ABC"[chosen]
                        model_answer_text = opt_list[chosen]
                    else:
                        letter = None
                        model_answer_text = None

                    out_chunks = []
                    for ch in chunks:
                        rs = ch.get("resampling_bias_importance") or 0.0
                        cf = ch.get("counterfactual_bias_importance") or 0.0
                        cfkl = ch.get("counterfactual_bias_kl") or 0.0
                        out_chunks.append({
                            **ch,
                            "summary": pretty_tag(ch.get("function_tags")),
                            "is_misaligned": (ch.get("p_biased") or 0) >= 0.5,
                            # --- aliases the website reads ---
                            "counterfactual_importance_category_kl": abs(rs),
                            "counterfactual_importance_kl": cfkl,
                            "counterfactual_importance_logodds": cf,
                            "resampling_importance_category_kl": abs(rs),
                        })

                    # Build the step->step graph the site needs from the
                    # experiment's real `depends_on` dependency field.
                    # Edge source->target weight = |Δ_RS| of the source chunk
                    # (closest available proxy for how much removing the source
                    # shifts downstream reasoning).
                    by_source = {}
                    for ch in chunks:
                        t = ch["chunk_idx"]
                        for s in (ch.get("depends_on") or []):
                            try:
                                s = int(s)
                            except (TypeError, ValueError):
                                continue
                            by_source.setdefault(s, []).append(t)
                    idx2chunk = {c["chunk_idx"]: c for c in chunks}
                    step_importance = []
                    for s in sorted(by_source):
                        src = idx2chunk.get(s)
                        if src is None:
                            continue
                        w = abs(src.get("resampling_bias_importance") or 0.0)
                        step_importance.append({
                            "source_chunk_idx": s,
                            "source_chunk_text": src.get("chunk", ""),
                            "target_impacts": [
                                {"target_chunk_idx": t, "importance_score": w}
                                for t in sorted(set(by_source[s]))
                            ],
                        })

                    imps = [abs(c.get("resampling_bias_importance") or 0)
                            for c in chunks]
                    order = sorted(range(len(chunks)),
                                   key=lambda i: imps[i], reverse=True)[:5]
                    top = [{"step_idx": out_chunks[i]["chunk_idx"],
                            "step_text": out_chunks[i]} for i in order]
                    summary = {
                        "scenario_idx": item_key,
                        "num_chunks": it.get("num_chunks", len(chunks)),
                        "avg_importance": sum(imps) / len(imps) if imps else 0,
                        "max_importance": max(imps) if imps else 0,
                        "min_importance": min(imps) if imps else 0,
                        "top_influential_steps": top,
                    }

                    cat = it.get("category", "")
                    # nickname = "#N  category (lang)" — unique within each bucket
                    n = counters.get(soltype, 1) - 1  # already incremented above
                    scenario = {
                        "scenario_id": item_key,
                        "original_item_id": iid,
                        "nickname": f"#{n} {cat} ({lang})",
                        "urgency_type": (f"{cat} | {lang.upper()} | {cond} | "
                                         f"polarity {it.get('question_polarity', '')} | "
                                         f"answer: {it.get('base_answer_type', '')}"),
                        "category": cat,
                        "context_condition": cond,
                        "question_polarity": it.get("question_polarity"),
                        "language": lang,
                        "context": meta.get("context"),
                        "question": meta.get("question"),
                        "options": {k: meta.get(k)
                                    for k in ("ans0", "ans1", "ans2")},
                        "answer": meta.get("answer"),
                        "target": meta.get("target"),
                        "other": meta.get("other"),
                        "base_is_biased": it.get("base_is_biased"),
                        # what the model actually answered
                        "model_answer_letter": letter,
                        "model_answer_text": model_answer_text,
                        "model_answer_type": it.get("base_answer_type"),
                        "model_is_biased": it.get("base_is_biased"),
                    }
                    base_out = {
                        "prompt": base.get("prompt", ""),
                        "solution": base.get("solution", ""),
                        "answer": meta.get("answer"),
                    }

                    outdir = WEB / model / soltype / item_key
                    outdir.mkdir(parents=True, exist_ok=True)
                    for name, data in [("scenario", scenario),
                                       ("chunks_labeled", out_chunks),
                                       ("summary", summary),
                                       ("step_importance", step_importance),
                                       ("base_solution", base_out)]:
                        json.dump(data, open(outdir / f"{name}.json", "w",
                                             encoding="utf-8"),
                                  ensure_ascii=False, indent=1)
                    n_items += 1
    print(f"Done. Wrote {n_items} items to {WEB}")


if __name__ == "__main__":
    convert()
