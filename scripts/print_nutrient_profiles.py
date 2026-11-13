#!/usr/bin/env python3
"""Print nutrient profiles of real and generated meals as a sanity check.

Reads `<out_dir>/data_exports/` (written by `cli/evaluate.py`) and prints, to stdout only,
a per-meal-type summary followed by detailed profiles of the first `--n_samples` real and
generated meals. `make eval` runs it through
`EvaluationManager.run_nutrient_profile_sanity_check` (src/pipeline_eval/evaluation.py) as

    python3 scripts/print_nutrient_profiles.py --out_dir outputs/runs/<RUN_NAME> --n_samples 3
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
import argparse

def load_evaluation_data(out_dir: str) -> Dict:
    """Load the cached evaluation data from the output directory."""
    out_path = Path(out_dir)
    exports_dir = out_path / "data_exports"
    
    # Load all the cached data
    data = {
        "real_meals": pd.read_pickle(exports_dir / "real_meals_by_type.pkl"),
        "generated_meals": pd.read_pickle(exports_dir / "generated_meals_by_type.pkl"),
        "clusters_real": pd.read_pickle(exports_dir / "clusters_by_meal.pkl"),
        "clusters_gen": pd.read_pickle(exports_dir / "gen_clusters_by_meal.pkl") if (exports_dir / "gen_clusters_by_meal.pkl").exists() else None,
        "W": np.load(exports_dir / "nutrient_matrix.npy"),
        "nutrient_names": json.loads((exports_dir / "nutrient_names.json").read_text()),
        "RDI": np.load(exports_dir / "RDI.npy"),
        "feature_names": json.loads((exports_dir / "feature_names.json").read_text()),
        "code2name": json.loads((exports_dir / "food_code_to_name.json").read_text()),
    }
    
    return data

def grams_to_nutrients(grams: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Convert grams to nutrients: (grams/100) @ W"""
    return (grams.astype(np.float32) / 100.0) @ W.astype(np.float32)

def calculate_rdi_coverage(nutrients: np.ndarray, RDI: np.ndarray, meal_fraction: float) -> np.ndarray:
    """Calculate RDI coverage for each nutrient."""
    target_rdi = RDI * meal_fraction
    coverage = nutrients / np.maximum(target_rdi, 1e-8)
    return coverage

def print_meal_nutrient_profile(meal_grams: np.ndarray, 
                               meal_type: str,
                               W: np.ndarray,
                               nutrient_names: List[str],
                               RDI: np.ndarray,
                               feature_names: List[str],
                               code2name: Dict[str, str],
                               meal_fraction_map: Dict[str, float],
                               meal_idx: int = 0) -> None:
    """Print detailed nutrient profile for a single meal."""
    
    print(f"\n{'='*80}")
    print(f"MEAL PROFILE: {meal_type} (Index {meal_idx})")
    print(f"{'='*80}")
    
    # Get meal fraction
    meal_frac = meal_fraction_map.get(meal_type, 0.33)
    print(f"Meal fraction: {meal_frac:.1%} of daily RDI")
    
    # Show foods present in meal
    print("\nFOODS IN MEAL:")
    print(f"{'Food Code':<12} {'Food Name':<30} {'Grams':<8}")
    print(f"{'-'*50}")
    
    total_grams = 0
    for i, grams in enumerate(meal_grams):
        if grams > 0.1:  # Only show foods with meaningful amounts
            food_code = feature_names[i]
            food_name = code2name.get(food_code, food_code)
            print(f"{food_code:<12} {food_name[:29]:<30} {grams:>7.1f}")
            total_grams += grams
    
    print(f"{'-'*50}")
    print(f"{'TOTAL':<42} {total_grams:>7.1f} g")
    
    # Calculate nutrients
    nutrients = grams_to_nutrients(meal_grams.reshape(1, -1), W)[0]
    target_rdi = RDI * meal_frac
    coverage = calculate_rdi_coverage(nutrients, RDI, meal_frac)
    
    # Calculate energy
    kcal_spec = {"Protein_g": 4, "Carb_g": 4, "Fat_g": 9}
    energy = 0
    for nutrient, kcal_per_g in kcal_spec.items():
        if nutrient in nutrient_names:
            idx = nutrient_names.index(nutrient)
            energy += nutrients[idx] * kcal_per_g
    
    print(f"\nENERGY: {energy:.1f} kcal")
    
    # Show nutrient breakdown
    print("\nNUTRIENT BREAKDOWN:")
    print(f"{'Nutrient':<25} {'Amount':<12} {'Target':<12} {'Coverage':<10} {'Status'}")
    print(f"{'-'*70}")
    
    # Group nutrients by category
    macronutrients = ["Protein_g", "Carb_g", "Fat_g", "Fiber_g", "Sugars_total_g", "SFA_g"]
    vitamins = [n for n in nutrient_names if "Vitamin" in n or "Folate" in n]
    minerals = [n for n in nutrient_names if any(m in n for m in ["Calcium", "Iron", "Zinc", "Sodium", "Potassium", "Phosphorus", "Magnesium"])]
    other = [n for n in nutrient_names if n not in macronutrients + vitamins + minerals]
    
    categories = [
        ("MACRONUTRIENTS", macronutrients),
        ("VITAMINS", vitamins),
        ("MINERALS", minerals),
        ("OTHER", other)
    ]
    
    for category_name, nutrient_list in categories:
        if not nutrient_list:
            continue
            
        print(f"\n{category_name}:")
        for nutrient in nutrient_list:
            if nutrient in nutrient_names:
                idx = nutrient_names.index(nutrient)
                amount = nutrients[idx]
                target = target_rdi[idx]
                cov = coverage[idx]
                
                # Format amounts appropriately
                if amount < 1:
                    amount_str = f"{amount:.3f}"
                else:
                    amount_str = f"{amount:.1f}"
                    
                if target < 1:
                    target_str = f"{target:.3f}"
                else:
                    target_str = f"{target:.1f}"
                
                # Status indicator
                if cov >= 0.8 and cov <= 1.2:
                    status = "✓ Good"
                elif cov >= 0.5 and cov <= 2.0:
                    status = "~ OK"
                else:
                    status = "✗ Poor"
                
                print(f"  {nutrient:<23} {amount_str:<12} {target_str:<12} {cov:>6.2f}x    {status}")
    
    # Calculate overall RDI deviation
    valid_coverage = coverage[np.isfinite(coverage) & (coverage > 0)]
    if len(valid_coverage) > 0:
        avg_deviation = np.mean(np.abs(valid_coverage - 1.0))
        print(f"\nOVERALL RDI DEVIATION: {avg_deviation:.3f}")
        print("(Lower is better; 0.0 = perfect, 1.0 = 100% average deviation)")

def print_comparison_summary(real_meals: Dict[str, np.ndarray],
                           gen_meals: Dict[str, np.ndarray],
                           W: np.ndarray,
                           nutrient_names: List[str],
                           RDI: np.ndarray,
                           meal_fraction_map: Dict[str, float],
                           n_samples: int = 3) -> None:
    """Print comparison summary between real and generated meals."""
    
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY: Real vs Generated Meals")
    print(f"{'='*80}")
    
    for meal_type in ["Breakfast", "Lunch", "Dinner"]:
        if meal_type not in real_meals or meal_type not in gen_meals:
            continue
            
        print(f"\n{meal_type.upper()}:")
        print(f"{'Metric':<25} {'Real':<15} {'Generated':<15} {'Difference'}")
        print(f"{'-'*70}")
        
        # Sample some meals for comparison
        real_sample = real_meals[meal_type][:n_samples]
        gen_sample = gen_meals[meal_type][:n_samples]
        
        # Calculate average nutrients
        real_nutrients = grams_to_nutrients(real_sample, W)
        gen_nutrients = grams_to_nutrients(gen_sample, W)
        
        real_avg = np.mean(real_nutrients, axis=0)
        gen_avg = np.mean(gen_nutrients, axis=0)
        
        # Calculate RDI coverage
        meal_frac = meal_fraction_map.get(meal_type, 0.33)
        real_coverage = real_avg / (RDI * meal_frac)
        gen_coverage = gen_avg / (RDI * meal_frac)
        
        # Calculate deviations
        real_dev = np.mean(np.abs(real_coverage - 1.0))
        gen_dev = np.mean(np.abs(gen_coverage - 1.0))
        
        print(f"{'Avg RDI Deviation':<25} {real_dev:<15.3f} {gen_dev:<15.3f} {gen_dev-real_dev:+.3f}")
        
        # Energy comparison
        kcal_spec = {"Protein_g": 4, "Carb_g": 4, "Fat_g": 9}
        real_energy = 0
        gen_energy = 0
        for nutrient, kcal_per_g in kcal_spec.items():
            if nutrient in nutrient_names:
                idx = nutrient_names.index(nutrient)
                real_energy += real_avg[idx] * kcal_per_g
                gen_energy += gen_avg[idx] * kcal_per_g
        
        print(f"{'Avg Energy (kcal)':<25} {real_energy:<15.1f} {gen_energy:<15.1f} {gen_energy-real_energy:+.1f}")
        
        # Total grams
        real_total = np.mean(np.sum(real_sample, axis=1))
        gen_total = np.mean(np.sum(gen_sample, axis=1))
        print(f"{'Avg Total Grams':<25} {real_total:<15.1f} {gen_total:<15.1f} {gen_total-real_total:+.1f}")

def main():
    parser = argparse.ArgumentParser(description="Print nutrient profiles for sanity checking")
    parser.add_argument("--out_dir", type=str, required=True, 
                       help="Output directory from evaluation run (contains data_exports/)")
    parser.add_argument("--n_samples", type=int, default=3,
                       help="Number of sample meals to print per meal type")
    parser.add_argument("--meal_types", nargs="+", default=["Breakfast", "Lunch", "Dinner"],
                       help="Meal types to analyze")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading evaluation data...")
    data = load_evaluation_data(args.out_dir)
    
    # Meal fraction mapping
    meal_fraction_map = {"Breakfast": 0.25, "Lunch": 0.35, "Dinner": 0.40}
    
    print("Loaded data:")
    print(f"  Real meals: {list(data['real_meals'].keys())}")
    print(f"  Generated meals: {list(data['generated_meals'].keys())}")
    print(f"  Nutrients: {len(data['nutrient_names'])}")
    print(f"  Foods: {len(data['feature_names'])}")
    
    # Print comparison summary first
    print_comparison_summary(
        data['real_meals'], 
        data['generated_meals'],
        data['W'],
        data['nutrient_names'],
        data['RDI'],
        meal_fraction_map,
        args.n_samples
    )
    
    # Print detailed profiles for sample meals
    for meal_type in args.meal_types:
        if meal_type not in data['real_meals'] or meal_type not in data['generated_meals']:
            print(f"\nSkipping {meal_type} - not found in data")
            continue
            
        print(f"\n{'='*80}")
        print(f"DETAILED PROFILES: {meal_type.upper()}")
        print(f"{'='*80}")
        
        # Print real meal profiles
        print("\nREAL MEALS:")
        real_meals = data['real_meals'][meal_type]
        for i in range(min(args.n_samples, len(real_meals))):
            print_meal_nutrient_profile(
                real_meals[i],
                meal_type,
                data['W'],
                data['nutrient_names'],
                data['RDI'],
                data['feature_names'],
                data['code2name'],
                meal_fraction_map,
                i
            )
        
        # Print generated meal profiles
        print("\nGENERATED MEALS:")
        gen_meals = data['generated_meals'][meal_type]
        for i in range(min(args.n_samples, len(gen_meals))):
            print_meal_nutrient_profile(
                gen_meals[i],
                meal_type,
                data['W'],
                data['nutrient_names'],
                data['RDI'],
                data['feature_names'],
                data['code2name'],
                meal_fraction_map,
                i
            )

if __name__ == "__main__":
    main()
