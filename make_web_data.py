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
import re
from pathlib import Path

ANSWER_RE = re.compile(
    r"(?:respuesta es|my answer is|the answer is|answer is|respuesta:|answer:)"
    r"\s*\[?\(?([ABC])", re.IGNORECASE)

EXP = Path("/home/andrew/Documents/docs/1-super-related-work/papers-manual/"
           "ai-safety-research/experiments/bias-thought-anchors")
WEB = Path("/home/andrew/Documents/docs/1-super-related-work/papers-manual/"
           "ai-safety-research/experiments/bias-thought-anchors.com/src/app/data")

TEMP = "temperature_0.6_top_p_0.95"
MODELS = ["qwen3p7-plus", "gemini-2.5-flash", "claude-sonnet-4-6"]
LANGS = ["es", "en"]
CONDS = ["ambig", "disambig"]


def pretty_tag(tags):
    return tags[0].replace("_", " ") if tags else "step"


def convert():
    n_items = 0
    for model in MODELS:
        for lang in LANGS:
            for cond in CONDS:
                ana = EXP / f"analysis/sesgo/{model}/{lang}/{cond}/results.json"
                if not ana.exists():
                    continue
                items = json.load(open(ana, encoding="utf-8"))
                soltype = f"{lang}_{cond}"
                for it in items:
                    iid = it["item_id"]
                    roll = EXP / ("bias_rollouts/sesgo/"
                                  f"{model}/{TEMP}/{lang}/{cond}/{iid}")
                    if not roll.exists():
                        print(f"  skip (no rollout): {model}/{lang}/{cond}/{iid}")
                        continue
                    meta = json.load(open(roll / "item.json", encoding="utf-8"))
                    base = json.load(open(roll / "base_solution.json",
                                          encoding="utf-8"))
                    chunks = it["labeled_chunks"]

                    # Parse the answer the model actually gave from the full
                    # solution text (more complete than base_solution.solution).
                    sol_text = ""
                    cj = roll / "chunks.json"
                    if cj.exists():
                        sol_text = json.load(open(cj, encoding="utf-8")).get(
                            "solution_text", "")
                    m = ANSWER_RE.search(sol_text)
                    letter = m.group(1).upper() if m else None
                    opts = {"A": meta.get("ans0"), "B": meta.get("ans1"),
                            "C": meta.get("ans2")}
                    model_answer_text = opts.get(letter) if letter else None

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
                        "scenario_idx": iid,
                        "num_chunks": it.get("num_chunks", len(chunks)),
                        "avg_importance": sum(imps) / len(imps) if imps else 0,
                        "max_importance": max(imps) if imps else 0,
                        "min_importance": min(imps) if imps else 0,
                        "top_influential_steps": top,
                    }

                    cat = it.get("category", "")
                    idx = iid.split("_")[-1]
                    scenario = {
                        "scenario_id": iid,
                        "nickname": f"{cat} {idx}",
                        "urgency_type": (f"{cat} | {cond} | "
                                         f"polarity {it.get('question_polarity', '')} | "
                                         f"base answer: {it.get('base_answer_type', '')}"),
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

                    outdir = WEB / model / soltype / iid
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
