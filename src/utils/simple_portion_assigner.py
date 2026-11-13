#!/usr/bin/env python3
"""
Rule-based WWEIA-style (group, subgroup) for a food code and food name.

wweia_group_and_subgroup() looks at the integer value of the code (ranges 1000-9999, e.g.
7000-7999 beverages, 7500-7599 alcoholic beverages) and a few name keywords (pizza, sandwich,
soup, bar, cracker, diet, coffee/tea, water, juice). Any other value, including every 8-digit
FNDDS food code, gives ("other", ""). Its only caller is
MultiMealPresenceOnlyCVAE._is_beverage_by_group_classification (src/utils/multi_meal_presence_cvae.py),
used by diversify_top_n when cli/run_multi_meal_cvae.py (`make cvae_train`) draws sanity samples.
Generated meals are portioned by Generator._portion_rdi_per_kcal in src/core/generator.py.
"""

def wweia_group_and_subgroup(food_code: str, food_name: str) -> tuple[str, str]:
    code = int(food_code) if food_code.isdigit() else -1
    name = (food_name or "").lower()

    if 1000 <= code <= 1999: return "milk_dairy", ""
    if 2000 <= code <= 2999: return "protein_foods", ""
    if 3000 <= code <= 3999:
        if "pizza" in name: return "mixed_dishes","pizza"
        if "sandwich" in name: return "mixed_dishes","sandwiches"
        if "soup" in name: return "mixed_dishes","soups"
        return "mixed_dishes",""
    if 4000 <= code <= 4999: return "grains",""
    if 5000 <= code <= 5999:
        if "bar" in name: return "snacks_sweets","snack_bars"
        if "cracker" in name: return "snacks_sweets","crackers"
        return "snacks_sweets",""
    if 6000 <= code <= 6399: return "fruits",""
    if 6400 <= code <= 6999: return "vegetables",""
    # put alcohol BEFORE the beverage catch-all
    if 7500 <= code <= 7599: return "alcoholic_beverages",""
    if 7000 <= code <= 7999:
        if "diet" in name: return "beverages","diet_beverages"
        if "coffee" in name or "tea" in name: return "beverages","coffee_tea"
        if "water" in name and "flavor" in name: return "beverages","flavored_water"
        if "water" in name: return "beverages","plain_water"
        if "juice" in name: return "beverages","juice"
        return "beverages","sweetened_beverages"
    if 8000 <= code <= 8099: return "fats_oils",""
    if 8400 <= code <= 8499: return "condiments_sauces",""
    if 8800 <= code <= 8899: return "sugars",""
    if 9000 <= code <= 9699: return "baby_foods",""
    if 9800 <= code <= 9999: return "other",""
    return "other",""
