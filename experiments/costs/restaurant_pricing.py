#!/usr/bin/env python3
"""
Portion-based restaurant pricing (S1 Table 16).

Used by experiments/analyze_meal_substitutions.py (with --use_restaurant_portion_pricing)
and experiments/substitution_foldchange/plot_nutrient_fold_changes.py (Figure 4A costs).

- Map food names to base labels used for menu-style portion pricing
- Provide per-item portion specs (grams per portion, USD per portion)
- Cap stackable portions by item type to avoid unrealistic quantities
- Estimate the menu cost of a meal vector (grams per food code)

A flat overhead (e.g. a plate fee) is passed explicitly via overhead_usd; the
substitution analysis prices meals with overhead_usd=0 and adds its own overhead.
"""

from __future__ import annotations

from typing import Dict, Tuple, Iterable
import numpy as np


def clean_name_display(food_code: str, code2name: Dict[str, str]) -> str:
    """Return a concise display string for a food code, similar to plotting logic.

    Prefers the last parenthetical in the long description when present.
    """
    import re as _re
    nm = code2name.get(food_code, food_code)
    m = _re.findall(r"\(([^)]+)\)", nm)
    return (m[-1] if m else nm)


def _friendly_base_label(name: str) -> str:
    n = name.lower().strip()
    if 'potato' in n and 'mashed' in n:
        return 'Mashed potatoes'
    if 'salisbury steak' in n:
        return 'Salisbury steak with gravy'
    if ('meat loaf' in n) or ('meatloaf' in n):
        return 'Beef/pork meatloaf with tomato sauce'
    if 'chicken breast' in n:
        return 'Baked skinless chicken breast'
    if 'peas and carrots' in n:
        return 'Steamed peas & carrots'
    if 'cabbage' in n and 'red' in n:
        return 'Shredded red cabbage'
    if 'hot dog' in n and 'bun' in n:
        return 'Beef hot dog on bun'
    if 'ketchup' in n:
        return 'Ketchup'
    if 'mayonnaise' in n or ('salad dressing' in n and 'mayonnaise' in n):
        return 'Mayonnaise'
    if 'bread' in n and 'whole wheat' in n:
        return 'Whole wheat bread'
    if 'fruit juice' in n:
        return 'Fruit juice'
    if ('yogurt' in n and 'liquid' in n) or ('drinkable' in n and 'yogurt' in n):
        return 'Drinkable yogurt'
    if ('yogurt' in n) and ('lowfat' in n or 'plain' in n):
        return 'Low-fat yogurt'
    if 'rice' in n and 'cooked' in n:
        return 'Steamed white rice'
    if ('bean' in n) and ('dried' in n or 'no added fat' in n or 'cooked' in n):
        return 'Cooked beans'
    if 'cherries' in n:
        return 'Cherries'
    return name.strip()


def portion_spec_for_label(base_label: str) -> Tuple[float, float]:
    """Return (grams_per_portion, price_per_portion_usd) for menu pricing."""
    n = base_label.lower()
    # Soups and fruit salads: model per-bowl/per-cup pricing explicitly
    if 'soup' in n:
        # Typical bowl ~ 350 g, menu price ~ $6
        return (350.0, 6.00)
    if 'fruit salad' in n:
        # Typical cup ~ 200 g, menu price ~ $4
        return (200.0, 4.00)
    if 'salisbury steak' in n:
        return (170.0, 13.00)
    if 'meatloaf' in n:
        return (170.0, 12.50)
    if 'chicken breast' in n:
        return (170.0, 12.00)
    if 'beef hot dog on bun' in n or ('hot dog' in n and 'bun' in n):
        return (120.0, 4.00)
    if 'whole wheat bread' in n or ('bread' in n and 'wheat' in n):
        return (56.0, 1.00)
    if 'steamed white rice' in n:
        return (158.0, 3.00)
    if 'steamed peas & carrots' in n or 'peas and carrots' in n:
        return (160.0, 3.50)
    if 'cooked beans' in n:
        return (170.0, 4.00)
    if 'shredded red cabbage' in n:
        return (70.0, 3.00)
    if 'cherries' in n:
        return (140.0, 4.00)
    if 'low-fat yogurt' in n:
        return (170.0, 3.00)
    if 'drinkable yogurt' in n:
        return (120.0, 2.50)
    if 'fruit juice' in n or 'juice' in n:
        return (240.0, 2.50)
    if 'ketchup' in n:
        return (17.0, 0.10)
    if 'mayonnaise' in n:
        return (14.0, 0.20)
    if 'iced coffee' in n or ('coffee' in n and 'iced' in n):
        return (240.0, 3.00)
    if 'water' in n:
        return (240.0, 0.00)
    # Generic side
    return (150.0, 3.00)


def max_portions_for_label(base_label: str) -> float:
    """Return hard cap on stackable portions for a base label."""
    n = base_label.lower()
    # Tight caps to avoid unrealistic multi-bowl salads/soups
    if 'soup' in n:
        return 1.0
    if 'fruit salad' in n:
        return 1.0
    if 'water' in n:
        return 0.0
    if 'fruit juice' in n or 'juice' in n:
        return 1.0
    if 'drinkable yogurt' in n or 'low-fat yogurt' in n:
        return 1.0
    if 'iced coffee' in n or 'coffee' in n:
        return 1.0
    if 'ketchup' in n or 'mayonnaise' in n:
        return 2.0
    if any(k in n for k in ['pizza', 'burrito', 'sandwich', 'submarine', 'burger', 'casserole', 'steak', 'meatloaf']):
        return 1.5
    if any(k in n for k in ['rice', 'beans', 'peas & carrots', 'peas and carrots', 'cabbage']):
        return 2.0
    return 3.0


def estimate_restaurant_cost(
    meal_vec: np.ndarray,
    feature_names: Iterable[str],
    code2name: Dict[str, str],
    overhead_usd: float = 0.0,
) -> float:
    """Estimate restaurant-style cost for a single meal vector (grams by feature).

    Prices are computed as (capped portions) * (price per portion) per item and
    summed, plus an optional flat overhead.
    """
    total = 0.0
    # Cross-item caps per meal: at most 1 soup bowl and 1 fruit salad serving total
    remaining_caps = {
        "soup": 1.0,
        "fruit_salad": 1.0,
    }
    for j, code in enumerate(feature_names):
        grams = float(meal_vec[j])
        if grams <= 0:
            continue
        # Use full name to derive base label for portion logic (avoid losing key nouns)
        code8 = str(code).zfill(8)
        full_name = code2name.get(code8, str(code))
        base = _friendly_base_label(full_name)
        g_per, price_per = portion_spec_for_label(base)
        portions = grams / max(g_per, 1e-6)
        # Per-item cap
        cap_item = max_portions_for_label(base)
        portions = min(portions, cap_item)
        # Cross-item category caps
        lname = str(full_name).lower()
        if "soup" in lname:
            rem = max(0.0, remaining_caps.get("soup", 0.0))
            take = min(portions, rem)
            remaining_caps["soup"] = max(0.0, rem - take)
            portions = take
        elif "fruit salad" in lname:
            rem = max(0.0, remaining_caps.get("fruit_salad", 0.0))
            take = min(portions, rem)
            remaining_caps["fruit_salad"] = max(0.0, rem - take)
            portions = take
        total += portions * price_per
    return float(round(total + float(overhead_usd), 2))


def estimate_restaurant_costs_for_meals(
    meals: np.ndarray,
    feature_names: Iterable[str],
    code2name: Dict[str, str],
    overhead_usd: float = 0.0,
) -> np.ndarray:
    """Price each row of `meals` with estimate_restaurant_cost."""
    costs = [
        estimate_restaurant_cost(m, feature_names, code2name, overhead_usd=overhead_usd)
        for m in meals
    ]
    return np.asarray(costs, dtype=float)


__all__ = [
    "clean_name_display",
    "portion_spec_for_label",
    "max_portions_for_label",
    "estimate_restaurant_cost",
    "estimate_restaurant_costs_for_meals",
]


