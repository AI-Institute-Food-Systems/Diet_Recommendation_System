"""
Food-nutrient and RDI tables on one canonical nutrient list.

load_harmonized_nutrition() reads the per-100 g food-nutrient table and the daily-target table,
maps both onto the 31 canonical nutrients in CANON_ORDER (energy_kcal, protein_g, ..., folate_mcg)
and puts them on the same units (vitamin D in IU; selenium, vitamins A, K and B-12 and folate in
mcg). cli/evaluate.py (`make eval`) calls it with data/mappings/food_to_nutrient.csv and
data/raw/food_patterns/healthy_US_style_food_pattern.csv, whose "2000 cal" column gives the
daily targets.
"""
from pathlib import Path

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def load_harmonized_nutrition(
    food_csv: str,
    rdi_csv: str,
    *,
    id_col: Optional[str] = None,        # food code column in food_csv; None: 'Food code' if present, else the first column
    meal_fraction_map: Dict[str, float] = {"Breakfast": 0.25, "Lunch": 0.35, "Dinner": 0.40},
    use_standard_dri_overrides: bool = False,  # set True to force e.g. Vit D=600 IU, B12=2.4 mcg
) -> Dict[str, object]:
    """
    Load and harmonise the food-nutrient table (food_csv) and the RDI table (rdi_csv).

    Nutrients absent from food_csv become zero columns; nutrients without a matching RDI row
    get NaN targets.

    Returns:
      {
        'feature_ids': np.ndarray (D,),  # food codes as 8-digit strings
        'nutrient_names': List[str] (P,),
        'W_per100g': np.ndarray (D, P)  # canonical units
        'RDI_per_day': np.ndarray (P,), # canonical units, aligned to nutrient_names
        'RDI_per_meal': Dict[str, np.ndarray]  # RDI_per_day x meal_fraction_map[meal]
      }
    """

    # 1) Accepted column/row names per canonical nutrient (food-table and RDI-table spellings)
    COLS = {
        "energy_kcal": ["Energy (kcal)", "Energy, kcal", "Energy_kcal", "Energy\n(kcal)"],
        "protein_g": ["Protein (g)", "Protein\n(g)"],
        "fat_g": ["Total fat (g)", "Fat (g)", "Total Fat\n(g)"],
        "carb_g": ["Carbohydrate (g)", "Carbohydrate\n(g)"],
        "fiber_g": ["Fiber, total dietary (g)", "Fiber (total dietary) (g)", "Dietary fiber (g)", "Fiber, total dietary\n(g)"],
        "sodium_mg": ["Sodium (mg)", "Sodium\n(mg)"],
        "sugars_total_g": ["Total sugars (g)", "Sugars, total (g)", "Sugars, total\n(g)"],

        "sfa_g": ["Fatty acids, total saturated (g)", "Saturated fat (g)", "Saturated fats (g)", "Fatty acids, total saturated\n(g)"],
        "mono_fat_g": ["Fatty acids, total monounsaturated (g)", "Monounsaturated fats (g)", "Fatty acids, total monounsaturated\n(g)"],
        "poly_fat_g": ["Fatty acids, total polyunsaturated (g)", "Polyunsaturated fats (g)", "Fatty acids, total polyunsaturated\n(g)"],
        "cholesterol_mg": ["Cholesterol (mg)", "Cholesterol\n(mg)"],

        "calcium_mg": ["Calcium (mg)", "Calcium\n(mg)"],
        "iron_mg": ["Iron (mg)", "Iron\n(mg)"],
        "magnesium_mg": ["Magnesium (mg)", "Magnesium\n(mg)"],
        "phosphorus_mg": ["Phosphorus (mg)", "Phosphorus\n(mg)"],
        "potassium_mg": ["Potassium (mg)", "Potassium\n(mg)"],
        "zinc_mg": ["Zinc (mg)", "Zinc\n(mg)"],
        "copper_mg": ["Copper (mg)", "Copper\n(mg)"],
        "selenium_mcg": ["Selenium (mcg)", "Selenium (µg)", "Selenium (ug)", "Selenium (mg)", "Selenium\n(mcg)"],  # mg→mcg

        "vitamin_a_mcg": ["Vitamin A, RAE (mcg)", "Vitamin A (mcg_RAE)", "Vitamin A (mg_RAE)", "Vitamin A (µg RAE)", "Vitamin A, RAE\n(mcg_RAE)"],
        "vitamin_e_mg": ["Vitamin E (alpha-tocopherol) (mg)", "Vitamin E (mg)", "Vitamin E\n(alpha-tocopherol)\n(mg)"],
        "vitamin_d_iu": ["Vitamin D (IU)", "Vitamin D (D2 + D3) (mcg)", "Vitamin D (mcg)", "Vitamin D\n(D2 + D3)\n(mcg)"],    # mcg→IU
        "vitamin_c_mg": ["Vitamin C (mg)", "Vitamin C\n(mg)"],
        "thiamin_mg": ["Thiamin (mg)", "Thiamine (mg)", "Thiamin\n(mg)"],
        "riboflavin_mg": ["Riboflavin (mg)", "Riboflavin\n(mg)"],
        "niacin_mg": ["Niacin (mg)", "Niacin\n(mg)"],
        "vitamin_b6_mg": ["Vitamin B-6 (mg)", "Vitamin B-6\n(mg)"],
        "vitamin_b12_mcg": ["Vitamin B-12 (mcg)", "Vitamin B12 (mcg)", "Vitamin B-12 (µg)", "Vitamin B-12\n(mcg)"],
        "choline_mg": ["Choline, total (mg)", "Choline (mg)", "Choline, total\n(mg)"],
        "vitamin_k_mcg": ["Vitamin K (phylloquinone) (mcg)", "Vitamin K (mcg)", "Vitamin K (mg)", "Vitamin K\n(phylloquinone)\n(mcg)"],
        "folate_mcg": ["Folate, DFE (mcg)", "Folate (mcg_DFE)", "Folate (mg_DFE)", "Folate, total\n(mcg)"],
    }

    CANON_ORDER = [
        "energy_kcal","protein_g","fat_g","carb_g","fiber_g","sodium_mg","sugars_total_g",
        "sfa_g","mono_fat_g","poly_fat_g","cholesterol_mg",
        "calcium_mg","iron_mg","magnesium_mg","phosphorus_mg","potassium_mg","zinc_mg","copper_mg","selenium_mcg",
        "vitamin_a_mcg","vitamin_e_mg","vitamin_d_iu","vitamin_c_mg","thiamin_mg","riboflavin_mg","niacin_mg","vitamin_b6_mg",
        "vitamin_b12_mcg","choline_mg","vitamin_k_mcg","folate_mcg",
    ]

    def find_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
        for name in options:
            if name in df.columns:
                return name
        return None

    # 2) Load CSVs
    foods = pd.read_csv(food_csv)
    rdi = pd.read_csv(rdi_csv)

    # pick id column (use 'Food code' if available, otherwise first col)
    if id_col is None:
        if 'Food code' in foods.columns:
            id_col = 'Food code'
        else:
            id_col = foods.columns[0]
    feature_ids = foods[id_col].astype(str).str.zfill(8).to_numpy()  # Ensure 8-digit format

    # 3) Food matrix (per 100 g, canonical units)
    harmonized_cols = {}
    for key in CANON_ORDER:
        src = find_col(foods, COLS[key])
        if src is None:
            harmonized_cols[key] = np.zeros(len(foods), dtype=np.float32)
            continue

        vals = foods[src].astype(float).to_numpy()

        # Unit fixes on FOOD side
        if key == "vitamin_d_iu":
            # prefer IU column; otherwise convert mcg→IU
            if src.lower().endswith("(mcg)") or " (mcg)" in src or src.lower().endswith("mcg"):
                vals = vals * 40.0
        if key == "selenium_mcg":
            # if Selenium column was mg, convert to mcg
            if "(mg)" in src or src.lower().endswith("mg"):
                vals = vals * 1000.0

        harmonized_cols[key] = vals.astype(np.float32)

    nutrient_names = CANON_ORDER.copy()
    W_per100g = np.column_stack([harmonized_cols[k] for k in nutrient_names]).astype(np.float32)  # (D, P)

    # 4) Build harmonized RDI vector 
    # The RDI CSV has nutrients as rows and calorie levels as columns
    # We'll use the "2000 cal" column by default
    calorie_col = "2000 cal"
    if calorie_col not in rdi.columns:
        # Fallback to first calorie column
        calorie_cols = [col for col in rdi.columns if "cal" in col.lower()]
        if calorie_cols:
            calorie_col = calorie_cols[0]
        else:
            raise ValueError("No calorie columns found in RDI CSV")
    
    # Set the first column as index to make nutrient lookup easier
    rdi_indexed = rdi.set_index(rdi.columns[0])
    
    RDI_vals = []
    for key in nutrient_names:
        # Find the nutrient row in the RDI data
        nutrient_row = None
        for col_options in COLS.get(key, []):
            # Try to find a matching row - use exact match first, then fuzzy
            for idx in rdi_indexed.index:
                if str(idx).lower() == col_options.lower():
                    nutrient_row = idx
                    break
            if nutrient_row:
                break
                
            # If no exact match, try fuzzy matching
            for idx in rdi_indexed.index:
                if col_options.lower() in str(idx).lower() or str(idx).lower() in col_options.lower():
                    nutrient_row = idx
                    break
            if nutrient_row:
                break
        
        if nutrient_row is None:
            RDI_vals.append(np.nan)
            continue

        v = float(rdi_indexed.loc[nutrient_row, calorie_col])

        # Unit/magnitude fixes on the RDI side
        if key == "vitamin_d_iu":
            # if file gives mcg, convert to IU; if already IU, keep
            if "(mcg)" in str(nutrient_row) or "mcg" in str(nutrient_row).lower():
                v = v * 40.0
        if key == "selenium_mcg":
            # many sheets mislabel mg but put mcg numbers; if truly mg with small decimals, convert; if big ints, assume mcg already
            if "(mg)" in str(nutrient_row) or "mg" in str(nutrient_row).lower():
                # heuristic: values < 10 are probably mg → convert to mcg; bigger values likely already mcg mislabeled
                v = v * 1000.0 if v < 10 else v
        if key == "vitamin_b12_mcg":
            # common 1000× error (e.g., 3400 instead of 3.4)
            if v > 100.0:
                v = v / 1000.0
        # Vitamin A/K/Folate often mislabeled as mg in headers but numbers are mcg; keep numeric as mcg
        if key == "vitamin_a_mcg":
            # RDI file headers say mg_RAE but values are actually in mcg_RAE
            if "(mg_RAE)" in str(nutrient_row) or "mg_RAE" in str(nutrient_row).lower():
                # Values are already in mcg, don't convert
                pass
            # If value is very large (>10000), likely in ng instead of mcg
            elif v > 10000:
                v = v / 1000.0
            # If value is very large (>1000), likely already in mcg but RDI file has wrong values
            elif v > 1000:
                v = v / 1000.0
        if key == "vitamin_k_mcg":
            # RDI file headers say mg but values are actually in mcg
            if "(mg)" in str(nutrient_row) or "mg" in str(nutrient_row).lower():
                # Values are already in mcg, don't convert
                pass
            # If value is very large (>10000), likely in ng instead of mcg
            elif v > 10000:
                v = v / 1000.0
            # If value is very large (>1000), likely already in mcg but RDI file has wrong values
            elif v > 1000:
                v = v / 1000.0
        if key == "folate_mcg":
            # RDI file headers say mg_DFE but values are actually in mcg_DFE
            if "(mg_DFE)" in str(nutrient_row) or "mg_DFE" in str(nutrient_row).lower():
                # Values are already in mcg, don't convert
                pass
            # If value is very large (>10000), likely in ng instead of mcg
            elif v > 10000:
                v = v / 1000.0
            # If value is very large (>1000), likely already in mcg but RDI file has wrong values
            elif v > 1000:
                v = v / 1000.0

        RDI_vals.append(v)

    RDI_per_day = np.array(RDI_vals, dtype=np.float32)

    # Optional: override with standard DRIs if desired
    if use_standard_dri_overrides:
        overrides = {
            "vitamin_d_iu": 600.0,   # 15 mcg
            "vitamin_b12_mcg": 2.4,
        }
        for k, v in overrides.items():
            if k in nutrient_names:
                RDI_per_day[nutrient_names.index(k)] = float(v)

    # 5) Per-meal vectors (aligned to nutrient_names)
    RDI_per_meal = {
        meal: (RDI_per_day * float(frac)).astype(np.float32)
        for meal, frac in meal_fraction_map.items()
    }

    return {
        "feature_ids": feature_ids,
        "nutrient_names": nutrient_names,
        "W_per100g": W_per100g,          # shape (D, P) in canonical units
        "RDI_per_day": RDI_per_day,      # shape (P,)
        "RDI_per_meal": RDI_per_meal,    # dict[str->(P,)]
    }

# Print a summary of the harmonised tables for the repository's inputs.
if __name__ == "__main__":
    _repo_root = Path(__file__).resolve().parents[2]
    data = load_harmonized_nutrition(
        str(_repo_root / "data" / "mappings" / "food_to_nutrient.csv"),
        str(_repo_root / "data" / "raw" / "food_patterns" / "healthy_US_style_food_pattern.csv"),
        use_standard_dri_overrides=False,  # keep the RDI table's targets
    )
    print("Nutrients:", data["nutrient_names"][:8], "...")
    print("W_per100g shape:", data["W_per100g"].shape)
    print("RDI (day) head:", data["RDI_per_day"][:8])
    for meal, vec in data["RDI_per_meal"].items():
        print(meal, "RDI head:", vec[:8])
