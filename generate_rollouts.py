"""
Generate chain-of-thought rollouts for bias items (SESGO / IberBBQ).

Supported reasoning models and providers
-----------------------------------------
Chat API with reasoning_content  (Fireworks):
  Qwen3.7 Plus       accounts/fireworks/models/qwen3p7-plus
  Qwen3.6 Plus       accounts/fireworks/models/qwen3p6-plus

Chat/thinking API  (Google):
  Gemini 2.5 Flash            gemini-2.5-flash

Chat/thinking API  (Anthropic):
  Claude Sonnet 4.6           claude-sonnet-4-6
  Claude Opus 4.8             claude-opus-4-8

Usage:
    python generate_rollouts.py \\
        --dataset sesgo --language es \\
        --categories genero racismo \\
        --context_condition ambig \\
        --num_items 50 --num_rollouts 30 \\
        --provider Fireworks --model accounts/fireworks/models/qwen3p7-plus

    python generate_rollouts.py \\
        --dataset sesgo --language es \\
        --provider Google --model gemini-2.5-flash

    python generate_rollouts.py \\
        --dataset sesgo --language es \\
        --provider Anthropic --model claude-sonnet-4-6

Output structure:
    bias_rollouts/{dataset}/{model}/temperature_{t}_top_p_{p}/{language}/{context_condition}/
        item_{idx}/
            item.json            # original bias item
            base_solution.json   # base CoT + answer + answer_type
            chunks.json          # sentence-level chunks of base CoT
            chunk_{i}/
                solutions.json   # rollout results for this chunk
"""

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import anthropic as _anthropic_sdk
import httpx
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from utils import split_solution_into_chunks

from data_utils import (
    extract_mcq_answer,
    format_prompt,
    get_answer_type,
    is_biased,
    load_esbbq_items,
    load_sesgo_items,
)
from prompts import SYSTEM_PROMPTS

load_dotenv()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Semaphore limiting concurrent Fireworks requests (2 in parallel stays under 36k Generated TPM)
_fw_semaphore: Optional[asyncio.Semaphore] = None

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Generate CoT rollouts for EsBBQ bias items")
parser.add_argument("--model", "-m", type=str,
                    default="accounts/fireworks/models/qwen3p7-plus",
                    help="Model identifier")
parser.add_argument("--provider", "-p", type=str, default="Fireworks",
                    choices=["Fireworks", "Google", "Anthropic"],
                    help="API provider")
parser.add_argument("--categories", "-c", nargs="*", default=None,
                    help="EsBBQ categories to include (default: all)")
parser.add_argument("--context_condition", type=str, default="ambig",
                    choices=["ambig", "disambig", "both"],
                    help="Filter by context condition")
parser.add_argument("--num_items", "-ni", type=int, default=50,
                    help="Number of EsBBQ items to process")
parser.add_argument("--num_rollouts", "-nr", type=int, default=30,
                    help="Number of rollouts per chunk")
parser.add_argument("--temperature", "-t", type=float, default=0.6)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_tokens", "-mt", type=int, default=4096)
parser.add_argument("--max_chunks", type=int, default=100,
                    help="Skip items whose CoT splits into more than this many chunks")
parser.add_argument("--output_dir", "-o", type=str, default="bias_rollouts")
parser.add_argument("--seed", "-s", type=int, default=44)
parser.add_argument("--force", "-f", action="store_true",
                    help="Regenerate even if output files exist")
parser.add_argument("--max_retries", type=int, default=3)
parser.add_argument("--thinking_budget", type=int, default=8000,
                    help="Token budget for extended thinking (Anthropic/Google). "
                         "Higher values produce longer CoTs with more chunks.")
parser.add_argument("--language", "-l", type=str, default="es",
                    choices=["es", "en"],
                    help="Language for model prompts: 'es' (Spanish) or 'en' (English)")
parser.add_argument("--dataset", "-d", type=str, default="esbbq",
                    choices=["esbbq", "sesgo"],
                    help="Bias dataset: 'esbbq' (Spain, IberBBQ) or 'sesgo' (Latin America)")
parser.add_argument("--use_local", action="store_true",
                    help="Load EsBBQ from local JSONL instead of HuggingFace (esbbq only)")
parser.add_argument("--local_data_dir", type=str,
                    default=str(Path(__file__).parent.parent /
                                "bias/IberBBQ/data_es"),
                    help="Path to local EsBBQ JSONL directory (esbbq only)")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

context_cond = None if args.context_condition == "both" else args.context_condition

model_short = args.model.split("/")[-1]
output_dir = (
    Path(args.output_dir)
    / args.dataset
    / model_short
    / f"temperature_{args.temperature}_top_p_{args.top_p}"
    / args.language
    / args.context_condition
)
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

def _build_api_input(item: Dict, system: str, language: str,
                     prefix: Optional[str] = None) -> Dict:
    """
    Build chat-style input dict for all providers.
    All 4 supported models use chat APIs — Qwen3 via Fireworks chat completions,
    Gemini via google-genai, Claude via anthropic SDK.

    When prefix is given (rollout resampling), it is appended to the user message
    so the model continues the partial chain of thought from that point.
    """
    user_text = format_prompt(item, language=language)
    if prefix:
        user_text += (
            "\n\n[Partial chain of thought so far — continue reasoning "
            "from here and then give your final answer]:\n" + prefix
        )
    return {"system": system, "user": user_text}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

async def _fireworks_qwen3_request(messages: Dict) -> Optional[str]:
    """
    Qwen3 on Fireworks via chat completions.
    Reasoning arrives as streaming `reasoning_content` deltas; the final answer
    arrives as `content` deltas. Both are combined into a single string separated
    by a </think> marker so downstream chunking logic works uniformly.
    """
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {"Authorization": f"Bearer {FIREWORKS_API_KEY}",
               "Content-Type": "application/json"}
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": messages["system"]},
            {"role": "user",   "content": messages["user"]},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "stream": True,
    }

    for attempt in range(args.max_retries):
        delay = 2 * (2 ** attempt)
        try:
            reasoning, answer = "", ""
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code == 429:
                        print(f"Fireworks rate limit (attempt {attempt+1}), waiting {delay}s…")
                        await asyncio.sleep(delay + random.uniform(0, 2))
                        continue
                    if resp.status_code != 200:
                        body = await resp.aread()
                        print(f"Fireworks error {resp.status_code}: {body[:200]}")
                        await asyncio.sleep(delay)
                        continue
                    async for line in resp.aiter_lines():
                        if not line.strip() or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            try:
                                chunk = json.loads(line[6:])
                                choices = chunk.get("choices") or []
                                delta = choices[0].get("delta", {}) if choices else {}
                                reasoning += delta.get("reasoning_content", "") or ""
                                answer    += delta.get("content", "") or ""
                            except json.JSONDecodeError:
                                pass
            return reasoning + "\n</think>\n" + answer
        except Exception as e:
            print(f"Fireworks request exception (attempt {attempt+1}): {e}")
            await asyncio.sleep(delay)
    return None


async def _google_request(messages: Dict) -> Optional[str]:
    """Gemini via google-genai SDK with thinking enabled."""
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise ImportError("Install google-genai: pip install google-genai")

    def _sync_call():
        client = genai.Client(api_key=GOOGLE_API_KEY)
        return client.models.generate_content(
            model=args.model,
            contents=f"{messages['system']}\n\n{messages['user']}",
            config=genai_types.GenerateContentConfig(
                thinking_config=genai_types.ThinkingConfig(
                    thinking_budget=args.thinking_budget,
                    include_thoughts=True,
                ),
                max_output_tokens=args.max_tokens,
            ),
        )

    for attempt in range(args.max_retries):
        delay = 2 * (2 ** attempt)
        try:
            response = await asyncio.to_thread(_sync_call)
            thinking, answer = [], []
            for part in response.candidates[0].content.parts:
                if getattr(part, "thought", None):
                    thinking.append(part.text)
                else:
                    answer.append(part.text)
            return "".join(thinking) + "\n</think>\n" + "".join(answer)
        except Exception as e:
            print(f"Google API error (attempt {attempt+1}): {e}")
            await asyncio.sleep(delay)
    return None


async def _anthropic_request(messages: Dict) -> Optional[str]:
    """Anthropic Claude extended-thinking request."""
    thinking_budget = args.thinking_budget
    max_tokens = thinking_budget + 2048

    client = _anthropic_sdk.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    for attempt in range(args.max_retries):
        delay = 2 * (2 ** attempt)
        try:
            response = await client.messages.create(
                model=args.model,
                max_tokens=max_tokens,
                thinking={"type": "enabled", "budget_tokens": thinking_budget},
                temperature=1,  # required for extended thinking
                system=messages["system"],
                messages=[{"role": "user", "content": messages["user"]}],
            )
            thinking = "".join(
                b.thinking for b in response.content if b.type == "thinking"
            )
            answer = "".join(
                b.text for b in response.content if b.type == "text"
            )
            return thinking + "\n</think>\n" + answer
        except _anthropic_sdk.RateLimitError:
            print(f"Anthropic rate limit (attempt {attempt+1}), waiting {delay}s…")
            await asyncio.sleep(delay)
        except Exception as e:
            print(f"Anthropic API error (attempt {attempt+1}): {e}")
            await asyncio.sleep(delay)
    return None


async def api_request(input_data: Dict) -> Optional[str]:
    """Dispatch to the right API based on --provider. Returns generated text or None."""
    if args.provider == "Fireworks":
        return await _fireworks_qwen3_request(input_data)
    if args.provider == "Google":
        return await _google_request(input_data)
    return await _anthropic_request(input_data)


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

async def generate_base_solution(item: Dict) -> Optional[Dict]:
    """Generate base CoT solution for one bias item."""
    input_data = _build_api_input(item, SYSTEM_PROMPTS[args.language], args.language)
    text = await api_request(input_data)
    if text is None:
        return None

    user_prompt = input_data["user"]
    letter, answer_idx = extract_mcq_answer(text, language=args.language)
    atype = get_answer_type(item, answer_idx)

    return {
        "prompt": user_prompt,
        "solution": text,
        "full_cot": user_prompt + "\n" + text,
        "answer": letter,
        "answer_idx": answer_idx,
        "answer_type": atype,
        "is_biased": is_biased(item, answer_idx),
    }


async def generate_rollout(item: Dict, chunk: str, full_prefix: str) -> Dict:
    """
    Generate one rollout by removing `chunk` from the CoT prefix and resampling.

    Mirrors thought-anchors' generate_rollout logic exactly.
    """
    prefix_without_chunk = full_prefix.replace(chunk, "").strip()

    input_data = _build_api_input(
        item, SYSTEM_PROMPTS[args.language], args.language,
        prefix=prefix_without_chunk or None,
    )
    text = await api_request(input_data)
    if text is None:
        return {"chunk_removed": chunk, "prefix_without_chunk": prefix_without_chunk,
                "error": "API failure"}

    # First sentence of the rollout is what the model regenerated for this chunk
    resampled_chunks = split_solution_into_chunks(text)
    chunk_resampled = resampled_chunks[0] if resampled_chunks else ""

    letter, answer_idx = extract_mcq_answer(text, language=args.language)
    atype = get_answer_type(item, answer_idx)

    user_prompt = input_data["user"]
    return {
        "chunk_removed": chunk,
        "prefix_without_chunk": prefix_without_chunk,
        "chunk_resampled": chunk_resampled,
        "rollout": text,
        "full_cot": user_prompt + "\n" + text,
        "answer": letter,
        "answer_idx": answer_idx,
        "answer_type": atype,
        "is_biased": is_biased(item, answer_idx),
    }


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------

async def process_item(item_idx: int, item: Dict) -> None:
    item_dir = output_dir / f"item_{item_idx}"
    item_dir.mkdir(exist_ok=True, parents=True)

    # Save item
    item_file = item_dir / "item.json"
    if not item_file.exists() or args.force:
        with open(item_file, "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2, ensure_ascii=False)

    # --- Base solution ---
    base_file = item_dir / "base_solution.json"
    base = None

    if base_file.exists() and not args.force:
        with open(base_file, encoding="utf-8") as f:
            base = json.load(f)
        print(f"[item {item_idx}] Loaded base solution → answer_type={base.get('answer_type')}")
    else:
        print(f"[item {item_idx}] Generating base solution…")
        base = await generate_base_solution(item)
        if base is None:
            print(f"[item {item_idx}] Failed to generate base solution, skipping.")
            return
        with open(base_file, "w", encoding="utf-8") as f:
            json.dump(base, f, indent=2, ensure_ascii=False)

    # --- Chunk the CoT ---
    chunks_file = item_dir / "chunks.json"
    if chunks_file.exists() and not args.force:
        with open(chunks_file, encoding="utf-8") as f:
            chunks_data = json.load(f)
        chunks = chunks_data["chunks"]
    else:
        # Extract CoT body for chunking: prefer thinking part before </think>;
        # fall back to answer part if thinking was empty (e.g. Claude short responses)
        solution_text = base["solution"]
        if "</think>" in solution_text:
            parts = solution_text.split("</think>", 1)
            cot_body = parts[0].strip() or parts[1].strip()
        else:
            cot_body = solution_text
        chunks = split_solution_into_chunks(cot_body)

        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump({"solution_text": solution_text, "chunks": chunks},
                      f, indent=2, ensure_ascii=False)

    if not chunks:
        print(f"[item {item_idx}] No chunks extracted, skipping.")
        return
    if len(chunks) > args.max_chunks:
        print(f"[item {item_idx}] Too many chunks ({len(chunks)}), skipping.")
        return

    print(f"[item {item_idx}] {len(chunks)} chunks → generating rollouts…")

    # Build cumulative prefixes (same as thought-anchors)
    cumulative_prefixes = []
    current = ""
    for chunk in chunks:
        current += chunk + " "
        cumulative_prefixes.append(current.strip())

    # --- Rollouts per chunk ---
    for chunk_idx, (chunk, full_prefix) in enumerate(zip(chunks, cumulative_prefixes)):
        chunk_dir = item_dir / f"chunk_{chunk_idx}"
        chunk_dir.mkdir(exist_ok=True, parents=True)

        solutions_file = chunk_dir / "solutions.json"
        existing = []
        if solutions_file.exists() and not args.force:
            with open(solutions_file, encoding="utf-8") as f:
                existing = json.load(f)

        valid_existing = [s for s in existing if "error" not in s and s.get("answer") is not None]
        needed = args.num_rollouts - len(valid_existing)

        if needed <= 0:
            print(f"  chunk {chunk_idx}: already have {len(valid_existing)} rollouts")
            continue

        print(f"  chunk {chunk_idx}: generating {needed} rollouts…")
        if args.provider == "Fireworks":
            async def _fw_rollout(i=item, c=chunk, p=full_prefix):
                async with _fw_semaphore:
                    return await generate_rollout(i, c, p)
            tasks = [_fw_rollout() for _ in range(needed)]
            new_solutions = list(await asyncio.gather(*tasks))
        else:
            tasks = [generate_rollout(item, chunk, full_prefix) for _ in range(needed)]
            new_solutions = list(await asyncio.gather(*tasks))

        # Only keep rollouts with a valid answer; filter API failures
        new_solutions = [s for s in new_solutions if "error" not in s]
        all_solutions = existing + new_solutions

        if not all_solutions:
            print(f"  chunk {chunk_idx}: all rollouts failed, will retry next run")
            continue

        chunk_dir.mkdir(exist_ok=True, parents=True)  # defensive re-create
        with open(solutions_file, "w", encoding="utf-8") as f:
            json.dump(all_solutions, f, indent=2, ensure_ascii=False)

        print(f"  chunk {chunk_idx}: saved {len(all_solutions)} total rollouts")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    global _fw_semaphore
    _fw_semaphore = asyncio.Semaphore(2)

    print(f"Loading {args.dataset.upper()} items "
          f"(lang={args.language}, condition={args.context_condition}, "
          f"categories={args.categories or 'all'})…")

    if args.dataset == "sesgo":
        items = load_sesgo_items(
            language=args.language,
            categories=args.categories,
            context_condition=context_cond,
            num_items=args.num_items,
            seed=args.seed,
        )
    else:
        items = load_esbbq_items(
            categories=args.categories,
            context_condition=context_cond,
            num_items=args.num_items,
            use_local=args.use_local,
            local_data_dir=args.local_data_dir,
            seed=args.seed,
        )

    print(f"Loaded {len(items)} items → output: {output_dir}")

    for item_idx, item in tqdm(items, desc="Processing items"):
        await process_item(item_idx, item)


if __name__ == "__main__":
    key_map = {
        "Fireworks": FIREWORKS_API_KEY,
        "Google":    GOOGLE_API_KEY,
        "Anthropic": ANTHROPIC_API_KEY,
    }
    if not key_map.get(args.provider):
        raise ValueError(f"{args.provider.upper()}_API_KEY not found in environment")

    asyncio.run(main())
