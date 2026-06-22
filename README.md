# Thought Anchors for Social Bias

**Which Reasoning Steps Matter in Extended-Thinking LLMs on Latin American Scenarios?**

Andres Mosquera-Hernandez · Universidad de Los Andes · `a.mosquerah2@uniandes.edu.co`

---

This repository contains the code and data for the paper **"Thought Anchors for Social Bias:
Which Reasoning Steps Matter in Extended Thinking LLMs on Latin American Scenarios"**
(GEBNLP @ ACL 2026 submission). We adapt the thought-anchor resampling framework of
[Bogdan et al. (2025)](https://arxiv.org/abs/2506.19143) to social bias measurement,
applying it to [SESGO](https://github.com/mvrobles/SESGO), a Latin American social bias
benchmark, across three extended-thinking frontier models.

The interactive visualization is available at `http://bias-thought-anchors-com.vercel.app`.

---

## Repository Structure

```
bias-thought-anchors/
├── sesgo_small/              # SESGO-small: 80-item stratified pilot (xlsx per category/lang)
├── bias_rollouts/            # Raw rollouts (gitignored if large)
│   └── sesgo/{model}/temperature_0.6_top_p_0.95/{lang}/{cond}/item_{n}/
│       ├── item.json         # Original SESGO item metadata
│       ├── base_solution.json # Prompt + base chain-of-thought
│       ├── chunks.json       # Sentence-level chunks of the base CoT
│       └── chunk_{i}/solutions.json  # K rollouts for position i
├── analysis/                 # Processed results (chunk scores, labels)
│   └── sesgo/{model}/{lang}/{cond}/
│       ├── results.json      # Per-item chunk-level importance scores + tags
│       └── category_summary.json
├── figures/                  # Generated paper figures (PDF + PNG)
├── paper/                    # ACL LaTeX source
│   ├── main.tex
│   ├── references.bib
│   └── main.pdf
├── generate_rollouts.py      # Step 1 — generate base CoT + K rollouts per chunk
├── analyze_rollouts.py       # Step 2 — compute importance metrics + label chunks
├── step_attribution.py       # Step 3 (optional) — pairwise step→step attribution
├── plots.py                  # Step 4 — generate paper figures
├── make_web_data.py          # Convert analysis output → visualizer JSON
├── stats_and_scatter.py      # Additional statistics and scatter plots
├── data_utils.py             # Dataset loading, pro-stereo classification
├── prompts.py                # Claude Haiku labeling prompt
├── utils.py                  # CoT chunking, embedding helpers
└── requirements.txt
```

---

## Method Overview

For each item in SESGO-small and each chunk $i$ of the base chain-of-thought:

1. Truncate the CoT prefix at chunk $i-1$.
2. Resample $K=3$ continuations from the same model.
3. Classify each rollout as *pro-stereo*, *anti-stereo*, or *unknown*.
4. Compute three importance metrics:
   - $\Delta_\text{RS}(i)$ — change in $P(\text{pro-stereo})$ across consecutive positions
   - $\Delta_\text{CF}(i)$ — counterfactual importance (semantically dissimilar rollouts only)
   - $D_\text{KL}(i)$ — KL divergence between dissimilar and baseline distributions
5. Label each chunk with a semantic function tag via Claude Haiku.

**SESGO-small**: 80 items × 3 models × K=3 rollouts = ~1,708 chunk-level scored rows and
3,023 labeled chunks.

---

## Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API keys go in `.env` (never commit this file):

```bash
FIREWORKS_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
```

**Estimated API cost to reproduce SESGO-small in full** (80 items, 3 models, K=3):
~$26 total ($6 Fireworks / $10 Google / $10 Anthropic including Haiku labeling).

---

## Reproducing the Paper Results

All commands assume you are in the `bias-thought-anchors/` directory with the venv active.

### Step 0 — Verify the environment

```bash
python test_smoke.py
```

Should exit without errors and print "smoke tests passed".

---

### Step 1 — Generate rollouts

Run once per model × language × condition combination. The SESGO-small pilot uses
`--num_items 5` per category (10 ambig + 10 disambig per category per language = 80 items).

**Qwen3.7-Plus (Fireworks)**

```bash
for LANG in es en; do
  for COND in ambig disambig; do
    python generate_rollouts.py \
      --dataset sesgo --language $LANG \
      --context_condition $COND \
      --num_items 5 --num_rollouts 3 \
      --provider Fireworks \
      --model accounts/fireworks/models/qwen3p7-plus \
      --temperature 0.6 --top_p 0.95 \
      --thinking_budget 8000
  done
done
```

**Gemini 2.5 Flash (Google)**

```bash
for LANG in es en; do
  for COND in ambig disambig; do
    python generate_rollouts.py \
      --dataset sesgo --language $LANG \
      --context_condition $COND \
      --num_items 5 --num_rollouts 3 \
      --provider Google \
      --model gemini-2.5-flash \
      --temperature 0.6 --top_p 0.95
  done
done
```

**Claude Sonnet 4.6 (Anthropic)**

```bash
for LANG in es en; do
  for COND in ambig disambig; do
    python generate_rollouts.py \
      --dataset sesgo --language $LANG \
      --context_condition $COND \
      --num_items 5 --num_rollouts 3 \
      --provider Anthropic \
      --model claude-sonnet-4-6 \
      --temperature 0.6 --top_p 0.95 \
      --thinking_budget 8000
  done
done
```

Output lands in `bias_rollouts/sesgo/{model}/temperature_0.6_top_p_0.95/{lang}/{cond}/`.

> **Note:** Fireworks enforces a generated-token rate limit that causes throttling;
> wall-clock time for the full pilot is approximately 48 hours across all three providers.

---

### Step 2 — Compute importance metrics and label chunks

Run once per model × language × condition. Claude Haiku (`claude-haiku-4-5-20251001`) is
called here for semantic labeling — this accounts for the majority of the Anthropic cost.

```bash
MODEL_SLUG=qwen3p7-plus  # or gemini-2.5-flash / claude-sonnet-4-6

for LANG in es en; do
  for COND in ambig disambig; do
    python analyze_rollouts.py \
      --rollouts_dir bias_rollouts/sesgo/${MODEL_SLUG}/temperature_0.6_top_p_0.95/${LANG}/${COND} \
      --output_dir analysis/sesgo/${MODEL_SLUG}/${LANG}/${COND} \
      --similarity_threshold 0.8 \
      --sentence_model all-MiniLM-L6-v2 \
      --claude_model claude-haiku-4-5-20251001
  done
done
```

Repeat for the other two model slugs. Output lands in `analysis/sesgo/{model}/{lang}/{cond}/results.json`.

---

### Step 3 — Generate paper figures

```bash
python plots.py
```

Saves five figures to `figures/` (PDF + PNG):

| File | Figure in paper |
|------|-----------------|
| `fig1_base_bias_rates.pdf` | Fig. 1 — Base pro-stereo rates by model and condition |
| `fig2_trajectory.pdf` | App. D — P(pro-stereo) along CoT position |
| `fig3_tag_importance.pdf` | Fig. 2 — Mean \|Δ_RS\| by semantic tag |
| `fig4_scatter.pdf` | App. D — Δ_RS vs. Δ_CF scatter |
| `fig5_heatmap_categories.pdf` | App. A — Pro-stereo rate by category × model |

---

### Step 4 — Compile the paper

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Requires a TeX Live installation with `texlive-fontsrecommended` and `texlive-fontsextra`
for Times and Helvetica fonts. To switch to review mode (anonymous, line numbers), change
line 5 of `main.tex` from `\usepackage{acl}` to `\usepackage[review]{acl}`.

---

### Step 5 (optional) — Update the interactive visualizer

```bash
python make_web_data.py
```

Converts `analysis/` and `bias_rollouts/` into the JSON format consumed by the
Next.js visualizer in `../bias-thought-anchors.com/`. Then:

```bash
cd ../bias-thought-anchors.com
npm install   # first time only
npm run dev   # open http://localhost:3000
```

---

## Key Results (SESGO-small Pilot)

| Finding | Value |
|---------|-------|
| Pro-stereo rate, ambiguous condition | 0% (all 3 models) |
| Pro-stereo rate, disambiguated condition | 33–75% (varies by model) |
| Total chunk-level importance scores | 1,708 |
| Total labeled chunks | 3,023 |
| High-importance chunks (\|Δ_RS\| > 0.2) | 37 (≈1.2%) |
| Most frequent tag among high-importance | `context_recall` (30%), `logical_reasoning` (24%) |
| `stereotype_activation` in high-importance | 0 (0%) |

All numbers are from the pilot dataset (80 items). 95% binomial CIs span ±30–45 pp;
treat them as illustrative rather than conclusive.

---

## Citation

```bibtex
@inproceedings{mosquera2026thoughtanchors,
  title     = {Thought Anchors for Social Bias: Which Reasoning Steps Matter
               in Extended Thinking {LLM}s on {L}atin {A}merican Scenarios},
  author    = {Mosquera-Hernandez, Andres},
  booktitle = {Proceedings of the 5th Workshop on Gender Bias in Natural
               Language Processing ({GeBNLP})},
  year      = {2026},
}
```

---

## License and Data Use

Code: MIT License.
Data: SESGO-small is derived from [SESGO](https://github.com/mvrobles/SESGO)
(Robles et al., 2024); please attribute the original authors when using this subset.
Chunk-level annotations and importance scores are released under CC BY 4.0.

> **Security note:** The `.env` file contains real API credentials.
> It is listed in `.gitignore` and must never be committed.
