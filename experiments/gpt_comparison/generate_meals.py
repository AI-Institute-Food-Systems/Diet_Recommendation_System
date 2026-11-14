#!/usr/bin/env python3
"""Generate meals for every archetype with the CVAE and with a GPT model (`make gpt-generate`).

For each of the 34 (meal type, archetype) pairs the trained CVAE generates --n_per_archetype meals
with the settings of `make eval`, and the GPT model is asked for the same number of meals in
batches of --batch_size. GPT sees only the foods the CVAE may use for that archetype (the
decoder's gate, by FNDDS code and name) and the same per-meal targets; it answers with food codes
and grams in a fixed JSON schema.

Needs the OpenAI API: set OPENAI_API_KEY in the environment or in a .env file at the repository
root (copy .env.example). Install the extra package with `pip install -r requirements-gpt.txt`.

Writes <out_dir>/experiment_results.json (both sets of meals, the raw GPT responses and the
settings) and <out_dir>/generate.log. Evaluate them with experiments/gpt_comparison/evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from experiments.gpt_comparison.common import (  # noqa: E402
    FIBER_MIN_G, MAX_BEVERAGE_GRAMS, MEAL_ENERGY_SPLIT, MEAL_NAMES, MIN_SOLID_ITEMS, PROTEIN_MIN_G,
    SODIUM_MAX_MG, allowed_foods, build_generator, load_env_file,
)

log = logging.getLogger("gpt_comparison")

SYSTEM_MESSAGE = (
    "You are a dietitian. The catalog is given as a dict:\n"
    "- names_by_code: { \"<8-digit code>\": \"<human-readable name>\" }\n\n"
    "Use ONLY codes present as keys of names_by_code.\n"
    "Output ONLY JSON matching the strict schema with items of {code, grams}.\n"
    "Grams must be numeric."
)


def batch_prompt(meal_idx: int, n_meals: int, names_by_code: Dict[str, str]) -> str:
    """User prompt asking for n_meals meals of one meal type from the archetype's foods."""
    energy_target = int(2000 * MEAL_ENERGY_SPLIT[meal_idx])
    bev_cap = MAX_BEVERAGE_GRAMS[meal_idx]
    rules = (
        f"NUTRITIONAL TARGETS: {energy_target} kcal ±5%; "
        f"protein ≥ {PROTEIN_MIN_G[meal_idx]}g; fiber ≥ {FIBER_MIN_G[meal_idx]}g; "
        f"sodium ≤ {SODIUM_MAX_MG[meal_idx]}mg; "
        f"added sugars ≤ 10% energy; SFA ≤ 10% energy; beverage energy ≤ 25%.\n"
        f"STRUCTURAL CONSTRAINTS: 2–4 items total, ≤1 beverage, "
        f"≥{MIN_SOLID_ITEMS[meal_idx]} solids. Max grams: solids 300g, beverages {bev_cap}g.\n"
        f"PORTION GUIDELINES: Vegetables 30–200g, Fruits 50–300g, Grains 15–100g, "
        f"Protein 15–150g, Dairy 30–500g, Beverages 120–{bev_cap}g."
    )
    return (
        f"Create {n_meals} diverse {MEAL_NAMES[meal_idx]} meals.\n"
        + rules + "\n\n"
        "CATALOG (code → name):\n"
        f"names_by_code = {json.dumps(names_by_code, separators=(',', ':'))}\n\n"
        "Use ONLY codes that are keys of names_by_code.\n"
        "Return JSON: {\"meals\":[{\"meal_id\":1,\"items\":[{\"code\":\"00112233\",\"grams\":123.4}]}]}\n"
        "Do not include names in the output; just `code` and `grams`."
    )


def response_schema(n_meals: int) -> dict:
    """Strict JSON schema: exactly n_meals meals of 2–4 items, each an 8-digit code and grams."""
    item = {
        "type": "object",
        "properties": {"code": {"type": "string", "pattern": "^\\d{8}$"},
                       "grams": {"type": "number", "minimum": 0.0, "maximum": 1000.0}},
        "required": ["code", "grams"],
        "additionalProperties": False,
    }
    meal = {
        "type": "object",
        "properties": {"meal_id": {"type": "integer", "minimum": 1, "maximum": n_meals},
                       "items": {"type": "array", "minItems": 2, "maxItems": 4, "items": item}},
        "required": ["meal_id", "items"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "batch_meals",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"meals": {"type": "array", "minItems": n_meals, "maxItems": n_meals, "items": meal}},
                "required": ["meals"],
                "additionalProperties": False,
            },
        },
    }


def parse_meals(content: str, n_meals: int, allowed: set) -> List[Optional[Dict[str, float]]]:
    """Grams per food code for meal_id 1..n_meals; None for a missing or empty meal.
    Codes outside the catalogue and grams outside (0, 1000] are dropped."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return [None] * n_meals
    by_id: Dict[int, Optional[Dict[str, float]]] = {}
    for meal in data.get("meals", []) if isinstance(data, dict) else []:
        if not isinstance(meal, dict) or "meal_id" not in meal or "items" not in meal:
            continue
        foods: Dict[str, float] = {}
        for it in meal["items"]:
            if not isinstance(it, dict) or it.get("code") not in allowed:
                continue
            try:
                g = float(it.get("grams"))
            except (TypeError, ValueError):
                continue
            if 0 < g <= 1000:
                foods[it["code"]] = foods.get(it["code"], 0.0) + g
        by_id[int(meal["meal_id"])] = foods or None
    return [by_id.get(i + 1) for i in range(n_meals)]


class GPTClient:
    """Chat Completions calls with retries on rate limits and batch splitting on length limits."""

    def __init__(self, model: str, temperature: float, top_p: float, max_retries: int = 5):
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("The openai package is missing: pip install -r requirements-gpt.txt")

        self.client = OpenAI()  # reads OPENAI_API_KEY
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        name = model.lower()
        # Reasoning models take reasoning_effort and no sampling parameters.
        self.reasoning = name.startswith(("o1", "o3", "o4", "gpt-5"))

    def _complete(self, prompt: str, n_meals: int) -> str:
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": prompt}],
            response_format=response_schema(n_meals),
            timeout=180,
        )
        if self.reasoning:
            kwargs.update(reasoning_effort="low", max_completion_tokens=8000)
        else:
            kwargs.update(temperature=self.temperature, top_p=self.top_p, max_completion_tokens=2000)
        for attempt in range(self.max_retries + 1):
            try:
                try:
                    resp = self.client.chat.completions.create(**kwargs)
                except Exception as e:  # models without structured outputs: plain JSON mode
                    if "json_schema" not in str(e).lower() and "strict" not in str(e).lower():
                        raise
                    log.warning("%s does not support JSON schema; using JSON mode", self.model)
                    kwargs["response_format"] = {"type": "json_object"}
                    resp = self.client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                log.debug("finish_reason=%s usage=%s", choice.finish_reason, getattr(resp, "usage", None))
                content = (choice.message.content or "").strip()
                if not content or choice.finish_reason == "length":
                    raise RuntimeError("MODEL_LENGTH_HIT")
                return content
            except Exception as e:
                msg = str(e).lower()
                if "model_length_hit" in msg:
                    raise
                if attempt < self.max_retries and any(t in msg for t in ("rate limit", "429", "rate_limit_exceeded")):
                    delay = 0.5 * (2 ** attempt) * (0.85 + 0.3 * random.random())
                    log.debug("rate-limited; retry %d after %.1fs", attempt + 1, delay)
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError("retries exhausted")

    def meals(self, meal_idx: int, n_meals: int, names_by_code: Dict[str, str]) -> List[dict]:
        """n_meals results ({success, foods, error, raw_response}); up to three attempts while
        none succeed, halving the batch when the response is cut off."""
        allowed = set(names_by_code)
        for attempt in range(3):
            try:
                content = self._complete(batch_prompt(meal_idx, n_meals, names_by_code), n_meals)
            except RuntimeError as e:
                if "MODEL_LENGTH_HIT" not in str(e):
                    raise
                if n_meals == 1:
                    return [{"success": False, "foods": None, "error": "response cut off", "raw_response": None}]
                half = n_meals // 2
                return self.meals(meal_idx, half, names_by_code) + self.meals(meal_idx, n_meals - half, names_by_code)
            parsed = parse_meals(content, n_meals, allowed)
            results = [{"success": p is not None, "foods": p, "error": None if p else "meal missing from response",
                        "raw_response": content if i == 0 else None} for i, p in enumerate(parsed)]
            if any(r["success"] for r in results) or attempt == 2:
                return results
            log.debug("no usable meal in the response; attempt %d", attempt + 2)
        return results


def cvae_meals(gen, pair_code: int, meal_idx: int, n: int, seed: int) -> List[Dict[str, float]]:
    """n CVAE meals for one pair, portioned as in `make eval`; grams per food code."""
    df = gen.generate(n=n, meal=MEAL_NAMES[meal_idx], pair_codes=[pair_code] * n, temperature=1.0,
                      nucleus_p=1.0, max_beverages=1, portion_strategy="rdi_per_kcal", min_grams=5,
                      jitter_sigma=0.15, return_tidy=False, seed=seed)
    X = df[gen.feature_names].to_numpy(dtype=float)
    return [{code: float(X[i, j]) for j, code in enumerate(gen.feature_names) if X[i, j] > 0} for i in range(len(X))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default="outputs/runs/eval_run", help="Run folder with data_exports/ from `make eval`.")
    ap.add_argument("--model_dir", default=None, help="Trained model (default: <run_dir>/presence_only_model).")
    ap.add_argument("--out_dir", default=None, help="Default: <run_dir>/gpt_comparison/<model>.")
    ap.add_argument("--model", default="gpt-4o", help="OpenAI chat model.")
    ap.add_argument("--n_per_archetype", type=int, default=100, help="Meals per archetype and method.")
    ap.add_argument("--batch_size", type=int, default=10, help="Meals requested per API call.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_archetypes", type=int, default=0, help="Use only the first N archetypes (0 = all), for a quick test.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    load_env_file(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, or export it.")

    run_dir = Path(args.run_dir)
    model_dir = Path(args.model_dir) if args.model_dir else run_dir / "presence_only_model"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "gpt_comparison" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "generate.log", mode="w")])

    client = GPTClient(args.model, args.temperature, args.top_p)
    gen, exports = build_generator(run_dir, model_dir, args.seed)
    code2name = exports["code2name"]
    pairs = sorted(gen.pair_table.items())
    if args.max_archetypes > 0:
        pairs = pairs[:args.max_archetypes]

    results = {
        "settings": {k: v for k, v in vars(args).items()} | {"model_dir": str(model_dir),
                                                          "n_features": len(gen.feature_names)},
        "cvae_meals": [],
        "gpt_meals": [],
    }
    n = args.n_per_archetype
    for pair_code, (meal_idx, cluster_id) in pairs:
        foods = allowed_foods(gen, exports, pair_code)
        names_by_code = {c: code2name.get(c, f"Food_{c}") for c in foods}
        log.info("pair %d (%s, cluster %d): %d allowed foods", pair_code, MEAL_NAMES[meal_idx], cluster_id, len(foods))
        meta = {"pair_code": pair_code, "meal_idx": meal_idx, "cluster_id": cluster_id}

        for i, f in enumerate(cvae_meals(gen, pair_code, meal_idx, n, args.seed + pair_code)):
            results["cvae_meals"].append(meta | {"sample_idx": i, "success": bool(f), "foods": f or None,
                                                 "error": None if f else "empty meal"})
        gpt: List[dict] = []
        for start in range(0, n, args.batch_size):
            gpt += client.meals(meal_idx, min(args.batch_size, n - start), names_by_code)
            time.sleep(1.0)
        for i, r in enumerate(gpt):
            results["gpt_meals"].append(meta | {"sample_idx": i} | r)
        log.info("  GPT: %d of %d meals usable", sum(r["success"] for r in gpt), len(gpt))

    path = out_dir / "experiment_results.json"
    path.write_text(json.dumps(results, indent=1))
    log.info("CVAE meals: %d, GPT meals: %d usable of %d; written to %s",
             sum(m["success"] for m in results["cvae_meals"]),
             sum(m["success"] for m in results["gpt_meals"]), len(results["gpt_meals"]), path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
