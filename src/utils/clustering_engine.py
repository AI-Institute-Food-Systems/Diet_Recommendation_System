"""
MealClusterEngine: meal features, clustering per meal type, cluster statistics and the
per-cluster training matrices for the CVAE. Used by cli/run_clustering.py (`make clustering`);
the figures are drawn by utils/clustering_viz.py.

Everything is written under `output_dir` (outputs/clustering in the pipeline): the feature cache,
<meal>_feature_matrix_unscaled.csv, <meal>_cluster_diagnosis_report.txt,
feature_contribution_report.csv, and <meal>/cluster_<id>_training_data.pkl (+ .units.json).
cli/run_clustering.py also saves the results as cached_clustering.pkl.
"""
import os, json, time
from datetime import datetime
import numpy as np
import sys
import pandas as pd
from tqdm import tqdm

from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler

# Make `src` importable when this module is loaded from the CLI scripts.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_root = os.path.dirname(_this_dir)
if _src_root not in sys.path:
    sys.path.append(_src_root)

from utils.meal_nutrition import calculate_meal_nutrition_from_df

import re

def _canon_item_label(c) -> str:
    """Food-code column label as a plain string: '12345678.0' -> '12345678'; others unchanged."""
    s = str(c).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", s):
        return s.split(".")[0]
    return s


class MealClusterEngine:
    """
    Data/ML pipeline: features → clustering → validation → profiling → exports.
    No plotting here.
    """
    def __init__(self, meal_data, nutrient_data, meal_types=None, output_dir=None):
        self.meal_data = meal_data
        self.nutrient_data = nutrient_data
        self.meal_types = meal_types or list(meal_data.keys())
        # All engine artifacts (feature caches, exports, reports) live under this directory
        self.output_dir = os.path.normpath(output_dir or "outputs/clustering")

        # results
        self.meal_type_features = {}
        self.meal_type_metadata = {}
        self.meal_type_clusters = {}
        self.meal_type_stats = {}
        self.significant_clusters = {}
        self.cluster_profiles = {}

        # Long-format food tables per meal type (see _create_meal_type_tables)
        self.meal_type_tables = {}
        self._create_meal_type_tables()

        # Feature-inclusion toggles (default: include everything)
        # Can be modified by callers before feature preparation
        self.include_transforms = True
        self.include_composition = True

    def _create_meal_type_tables(self):
        """
        Build meal_type_tables: one row per non-zero food in each meal, with
        - Food code: The food identifier
        - grams: Amount of food in grams
        - meal_idx: Index of the meal this food belongs to
        The pipeline only logs their size; no later step reads them.
        """
        for meal_type in self.meal_types:
            if meal_type not in self.meal_data:
                continue
                
            meal_df = self.meal_data[meal_type]
            table_rows = []
            
            # Iterate through each meal (row)
            for meal_idx, meal_row in meal_df.iterrows():
                # Find foods with non-zero amounts
                non_zero_foods = meal_row[meal_row > 0]
                
                # Create a row for each food in this meal
                for food_code, grams in non_zero_foods.items():
                    table_rows.append({
                        'Food code': food_code,
                        'grams': grams,
                        'meal_idx': meal_idx
                    })
            
            # Create DataFrame for this meal type
            if table_rows:
                self.meal_type_tables[meal_type] = pd.DataFrame(table_rows)
                print(f"Created meal_type_tables[{meal_type}] with {len(table_rows)} food entries")
            else:
                print(f"Warning: No food entries found for {meal_type}")

    # ===================== FEATURE ENGINEERING =====================

    def _extract_nutritional_features(self, meal, meal_nutrition):
        """
        Nutrient-only features of one meal (`--feature-type nutritional`; the pipeline
        uses the hybrid features).
        """
        # Get foods present in meal
        present_foods = meal[meal > 0]
        
        features = {}
        
        # Macronutrients (g)
        features['protein_g'] = meal_nutrition.get('Protein (g)', 0)
        features['carbohydrate_g'] = meal_nutrition.get('Carbohydrate (g)', 0)
        features['fat_g'] = meal_nutrition.get('Total Fat (g)', 0)
        features['fiber_g'] = meal_nutrition.get('Fiber (total dietary) (g)', 0)
        
        # Micronutrients
        features['vitamin_d_iu'] = meal_nutrition.get('Vitamin D (IU)', 0)
        features['iron_mg'] = meal_nutrition.get('Iron (mg)', 0)
        features['calcium_mg'] = meal_nutrition.get('Calcium (mg)', 0)
        features['vitamin_c_mg'] = meal_nutrition.get('Vitamin C (mg)', 0)
        features['vitamin_a_iu'] = meal_nutrition.get('Vitamin A (IU)', 0)
        features['vitamin_b12_mcg'] = meal_nutrition.get('Vitamin B-12 (mcg)', 0)
        features['folate_mcg'] = meal_nutrition.get('Folate (mcg)', 0)
        features['zinc_mg'] = meal_nutrition.get('Zinc (mg)', 0)
        features['magnesium_mg'] = meal_nutrition.get('Magnesium (mg)', 0)
        features['potassium_mg'] = meal_nutrition.get('Potassium (mg)', 0)
        
        # Energy and ratios
        features['energy_kcal'] = meal_nutrition.get('Energy (kcal)', 0)
        features['protein_ratio'] = features['protein_g'] / max(features['energy_kcal'], 1) * 4  # 4 kcal/g
        features['carb_ratio'] = features['carbohydrate_g'] / max(features['energy_kcal'], 1) * 4
        features['fat_ratio'] = features['fat_g'] / max(features['energy_kcal'], 1) * 9  # 9 kcal/g
        
        # Food composition features
        features['food_diversity'] = len(present_foods)
        features['avg_portion_size'] = present_foods.mean() if len(present_foods) > 0 else 0
        
        return features
    
    def _extract_hybrid_features(self, meal, meal_nutrition):
        """
        Hybrid features of one meal: nutrients plus WWEIA food categories (84 features with
        include_transforms and include_composition on, as in the pipeline):
        - 8 nutrient features: protein, carbohydrate, fat and fiber grams, energy, kcal ratios
        - 5 log1p transforms (include_transforms)
        - 4 binned levels (protein, carb, fat, energy) and 3 macro balance ratios
        - 53 `grams_<category>` features (15 main categories, 38 subcategories); despite the
          name these are counts of the meal's foods in each category
          (see _categorize_foods_comprehensive)
        - 11 composition features (include_composition): macro diversity, meal balance score,
          category diversity, ingredient count, portion variability, nutritional balance,
          calorie density and the grain/vegetable/fruit/dairy shares of the category counts;
          the protein-foods share overwrites the kcal-based protein_ratio
        """
        # Get foods present in meal
        present_foods = meal[meal > 0]
        food_codes = [code for code, portion in meal.items() if portion > 0]
        
        features = {}
        
        # === NUTRITIONAL FEATURES (Core) ===
        # Macronutrients (g) - with log transformation for skewed features
        features['protein_g'] = meal_nutrition.get('Protein (g)', 0)
        features['carbohydrate_g'] = meal_nutrition.get('Carbohydrate (g)', 0)
        features['fat_g'] = meal_nutrition.get('Total Fat (g)', 0)
        features['fiber_g'] = meal_nutrition.get('Fiber (total dietary) (g)', 0)
        
        # Energy and ratios
        features['energy_kcal'] = meal_nutrition.get('Energy (kcal)', 0)
        features['protein_ratio'] = features['protein_g'] / max(features['energy_kcal'], 1) * 4
        features['carb_ratio'] = features['carbohydrate_g'] / max(features['energy_kcal'], 1) * 4
        features['fat_ratio'] = features['fat_g'] / max(features['energy_kcal'], 1) * 9
        
        # === TRANSFORMED NUTRITIONAL FEATURES ===
        # Log transformations for highly skewed features (add small constant to avoid log(0))
        if self.include_transforms:
            features['log_protein_g'] = np.log1p(features['protein_g'])  # log(1+x)
            features['log_carbohydrate_g'] = np.log1p(features['carbohydrate_g'])
            features['log_fat_g'] = np.log1p(features['fat_g'])
            features['log_energy_kcal'] = np.log1p(features['energy_kcal'])
            features['log_fiber_g'] = np.log1p(features['fiber_g'])
        
        # Binned nutritional features (categorical ranges)
        features['protein_level'] = self._bin_nutritional_value(features['protein_g'], 'protein')
        features['carb_level'] = self._bin_nutritional_value(features['carbohydrate_g'], 'carb')
        features['fat_level'] = self._bin_nutritional_value(features['fat_g'], 'fat')
        features['energy_level'] = self._bin_nutritional_value(features['energy_kcal'], 'energy')
        
        # === NUTRITIONAL INTERACTION FEATURES ===
        # Protein-carb balance, protein-fat balance, etc.
        features['protein_carb_balance'] = features['protein_g'] / max(features['carbohydrate_g'], 1)
        features['protein_fat_balance'] = features['protein_g'] / max(features['fat_g'], 1)
        features['carb_fat_balance'] = features['carbohydrate_g'] / max(features['fat_g'], 1)
        
        # Macronutrient diversity (how balanced are the macros)
        if self.include_composition:
            macro_values = [features['protein_g'], features['carbohydrate_g'], features['fat_g']]
            features['macro_diversity'] = np.std(macro_values) / max(np.mean(macro_values), 1)
        
        # === FOOD CATEGORY FEATURES (Comprehensive) ===
        # Use comprehensive food categorization with all WWEIA categories
        food_groups = self._categorize_foods_comprehensive(food_codes)
        
        # === MEAL COMPOSITION PATTERN FEATURES ===
        # These features help distinguish between similar meals by looking at HOW foods are combined
        if self.include_composition:
            # Meal balance score (how evenly distributed are the food groups)
            active_groups = [g for g in food_groups.values() if g > 0]
            if len(active_groups) > 1:
                features['meal_balance_score'] = 1 - (np.std(active_groups) / max(np.mean(active_groups), 1))
            else:
                features['meal_balance_score'] = 0
        
        # Food category diversity (number of main categories and subcategories present)
        if self.include_composition:
            features['food_category_diversity'] = len([g for g in food_groups.values() if g > 0])
        
        # === CATEGORY FEATURES (named grams_*, but they hold food counts) ===
        # Main categories
        features['grams_milk_dairy'] = food_groups.get('milk_dairy', 0)
        features['grams_protein_foods'] = food_groups.get('protein_foods', 0)
        features['grams_mixed_dishes'] = food_groups.get('mixed_dishes', 0)
        features['grams_grains'] = food_groups.get('grains', 0)
        features['grams_snacks_sweets'] = food_groups.get('snacks_sweets', 0)
        features['grams_fruits'] = food_groups.get('fruits', 0)
        features['grams_vegetables'] = food_groups.get('vegetables', 0)
        features['grams_beverages'] = food_groups.get('beverages', 0)
        features['grams_alcoholic_beverages'] = food_groups.get('alcoholic_beverages', 0)
        features['grams_water'] = food_groups.get('water', 0)
        features['grams_fats_oils'] = food_groups.get('fats_oils', 0)
        features['grams_condiments_sauces'] = food_groups.get('condiments_sauces', 0)
        features['grams_sugars'] = food_groups.get('sugars', 0)
        features['grams_baby_foods'] = food_groups.get('baby_foods', 0)
        features['grams_other'] = food_groups.get('other', 0)
        
        # Subcategories
        features['grams_milk'] = food_groups.get('milk', 0)
        features['grams_flavored_milk'] = food_groups.get('flavored_milk', 0)
        features['grams_dairy_drinks'] = food_groups.get('dairy_drinks', 0)
        features['grams_cheese'] = food_groups.get('cheese', 0)
        features['grams_yogurt'] = food_groups.get('yogurt', 0)
        features['grams_meats'] = food_groups.get('meats', 0)
        features['grams_poultry'] = food_groups.get('poultry', 0)
        features['grams_seafood'] = food_groups.get('seafood', 0)
        features['grams_eggs'] = food_groups.get('eggs', 0)
        features['grams_cured_meats'] = food_groups.get('cured_meats', 0)
        features['grams_plant_proteins'] = food_groups.get('plant_proteins', 0)
        features['grams_mixed_meat_dishes'] = food_groups.get('mixed_meat_dishes', 0)
        features['grams_mixed_bean_dishes'] = food_groups.get('mixed_bean_dishes', 0)
        features['grams_mixed_grain_dishes'] = food_groups.get('mixed_grain_dishes', 0)
        features['grams_asian_dishes'] = food_groups.get('asian_dishes', 0)
        features['grams_mexican_dishes'] = food_groups.get('mexican_dishes', 0)
        features['grams_pizza'] = food_groups.get('pizza', 0)
        features['grams_sandwiches'] = food_groups.get('sandwiches', 0)
        features['grams_soups'] = food_groups.get('soups', 0)
        features['grams_cooked_grains'] = food_groups.get('cooked_grains', 0)
        features['grams_breads_rolls'] = food_groups.get('breads_rolls', 0)
        features['grams_quick_breads'] = food_groups.get('quick_breads', 0)
        features['grams_cereals'] = food_groups.get('cereals', 0)
        features['grams_savory_snacks'] = food_groups.get('savory_snacks', 0)
        features['grams_crackers'] = food_groups.get('crackers', 0)
        features['grams_snack_bars'] = food_groups.get('snack_bars', 0)
        features['grams_sweet_bakery'] = food_groups.get('sweet_bakery', 0)
        features['grams_candy'] = food_groups.get('candy', 0)
        features['grams_desserts'] = food_groups.get('desserts', 0)
        features['grams_juice'] = food_groups.get('juice', 0)
        features['grams_diet_beverages'] = food_groups.get('diet_beverages', 0)
        features['grams_sweetened_beverages'] = food_groups.get('sweetened_beverages', 0)
        features['grams_coffee_tea'] = food_groups.get('coffee_tea', 0)
        features['grams_plain_water'] = food_groups.get('plain_water', 0)
        features['grams_flavored_water'] = food_groups.get('flavored_water', 0)
        features['grams_baby_beverages'] = food_groups.get('baby_beverages', 0)
        features['grams_infant_formulas'] = food_groups.get('infant_formulas', 0)
        features['grams_human_milk'] = food_groups.get('human_milk', 0)
        
        # === ENHANCED COMPOSITION FEATURES ===
        if self.include_composition:
            # Ingredient complexity
            features['ingredient_count'] = len(present_foods)
            features['portion_variability'] = (
                present_foods.std() / max(present_foods.mean(), 1) if len(present_foods) > 1 else 0
            )
            
            # Nutritional balance score (0-1)
            features['nutritional_balance'] = self._calculate_nutritional_balance(meal_nutrition)
            
            # Calorie density
            total_weight = sum(present_foods)
            features['calorie_density'] = features['energy_kcal'] / max(total_weight, 1)
        
        # === ADDITIONAL COMPOSITION FEATURES ===
        # Food group shares of the category counts (main + subcategories)
        if self.include_composition:
            total_grams = sum([g for g in food_groups.values() if g > 0])
            if total_grams > 0:
                # Note: this overwrites the kcal-based 'protein_ratio' set above
                features['protein_ratio'] = food_groups.get('protein_foods', 0) / total_grams
                features['grain_ratio'] = food_groups.get('grains', 0) / total_grams
                features['vegetable_ratio'] = food_groups.get('vegetables', 0) / total_grams
                features['fruit_ratio'] = food_groups.get('fruits', 0) / total_grams
                features['dairy_ratio'] = food_groups.get('milk_dairy', 0) / total_grams
            else:
                features['protein_ratio'] = 0
                features['grain_ratio'] = 0
                features['vegetable_ratio'] = 0
                features['fruit_ratio'] = 0
                features['dairy_ratio'] = 0
        
        return features


    def _categorize_foods_comprehensive(self, food_codes):
        """
        Count the given foods per WWEIA main category and subcategory, using each food's
        WWEIA Category code from data/mappings/food_ingredient_map.pkl (loaded once and kept
        on the engine). Codes without a mapping count as 'other'.
        """
        food_groups = {}
        
        # Initialize all main categories and subcategories
        # Main categories (15)
        food_groups.update({
            'milk_dairy': 0, 'protein_foods': 0, 'mixed_dishes': 0, 'grains': 0, 'snacks_sweets': 0,
            'fruits': 0, 'vegetables': 0, 'beverages': 0, 'alcoholic_beverages': 0, 'water': 0,
            'fats_oils': 0, 'condiments_sauces': 0, 'sugars': 0, 'baby_foods': 0, 'other': 0
        })
        
        # Subcategories (40+)
        food_groups.update({
            'milk': 0, 'flavored_milk': 0, 'dairy_drinks': 0, 'cheese': 0, 'yogurt': 0,
            'meats': 0, 'poultry': 0, 'seafood': 0, 'eggs': 0, 'cured_meats': 0, 'plant_proteins': 0,
            'mixed_meat_dishes': 0, 'mixed_bean_dishes': 0, 'mixed_grain_dishes': 0, 'asian_dishes': 0,
            'mexican_dishes': 0, 'pizza': 0, 'sandwiches': 0, 'soups': 0, 'cooked_grains': 0,
            'breads_rolls': 0, 'quick_breads': 0, 'cereals': 0, 'savory_snacks': 0, 'crackers': 0,
            'snack_bars': 0, 'sweet_bakery': 0, 'candy': 0, 'desserts': 0, 'juice': 0,
            'diet_beverages': 0, 'sweetened_beverages': 0, 'coffee_tea': 0, 'plain_water': 0,
            'flavored_water': 0, 'baby_beverages': 0, 'infant_formulas': 0, 'human_milk': 0
        })
        
        # Load the food ingredient mapping if not already loaded
        if not hasattr(self, '_food_code_to_wweia_mapping'):
            try:
                mapping_df = pd.read_pickle('data/mappings/food_ingredient_map.pkl')
                # Create mapping from food code to WWEIA Category code
                self._food_code_to_wweia_mapping = dict(zip(
                    mapping_df['Food code'].astype(str), 
                    mapping_df['WWEIA Category code']
                ))
                print(f"Loaded food code mapping: {len(self._food_code_to_wweia_mapping)} food codes mapped to WWEIA categories")
            except Exception as e:
                print(f"Warning: Could not load food ingredient mapping: {e}")
                self._food_code_to_wweia_mapping = {}
        
        for food_code in food_codes:
            if pd.isna(food_code):
                continue
                
            # Convert food code to string for lookup
            food_code_str = str(int(float(food_code)))
            
            # Look up the WWEIA Category code for this food code
            wweia_code = self._food_code_to_wweia_mapping.get(food_code_str)
            
            if wweia_code is None:
                # If no mapping found, categorize as 'other'
                food_groups['other'] += 1
                continue
            
            # WWEIA category-code ranges
            if 1000 <= wweia_code < 2000:
                food_groups['milk_dairy'] += 1
                # Subcategories
                if 1002 <= wweia_code < 1200:
                    food_groups['milk'] += 1
                elif 1200 <= wweia_code < 1300:
                    food_groups['flavored_milk'] += 1
                elif 1400 <= wweia_code < 1500:
                    food_groups['dairy_drinks'] += 1
                elif 1600 <= wweia_code < 1700:
                    food_groups['cheese'] += 1
                elif 1800 <= wweia_code < 1900:
                    food_groups['yogurt'] += 1
                    
            elif 2000 <= wweia_code < 3000:
                food_groups['protein_foods'] += 1
                # Subcategories
                if 2000 <= wweia_code < 2100:
                    food_groups['meats'] += 1
                elif 2200 <= wweia_code < 2300:
                    food_groups['poultry'] += 1
                elif 2400 <= wweia_code < 2500:
                    food_groups['seafood'] += 1
                elif 2500 <= wweia_code < 2600:
                    food_groups['eggs'] += 1
                elif 2600 <= wweia_code < 2700:
                    food_groups['cured_meats'] += 1
                elif 2800 <= wweia_code < 2900:
                    food_groups['plant_proteins'] += 1
                    
            elif 3000 <= wweia_code < 4000:
                food_groups['mixed_dishes'] += 1
                # Subcategories
                if 3000 <= wweia_code < 3100:
                    food_groups['mixed_meat_dishes'] += 1
                elif 3100 <= wweia_code < 3200:
                    food_groups['mixed_bean_dishes'] += 1
                elif 3200 <= wweia_code < 3300:
                    food_groups['mixed_grain_dishes'] += 1
                elif 3400 <= wweia_code < 3500:
                    food_groups['asian_dishes'] += 1
                elif 3500 <= wweia_code < 3600:
                    food_groups['mexican_dishes'] += 1
                elif 3600 <= wweia_code < 3700:
                    food_groups['pizza'] += 1
                elif 3700 <= wweia_code < 3800:
                    food_groups['sandwiches'] += 1
                elif 3800 <= wweia_code < 3900:
                    food_groups['soups'] += 1
                    
            elif 4000 <= wweia_code < 5000:
                food_groups['grains'] += 1
                # Subcategories
                if 4000 <= wweia_code < 4100:
                    food_groups['cooked_grains'] += 1
                elif 4200 <= wweia_code < 4300:
                    food_groups['breads_rolls'] += 1
                elif 4400 <= wweia_code < 4500:
                    food_groups['quick_breads'] += 1
                elif 4600 <= wweia_code < 4700:
                    food_groups['cereals'] += 1
                elif 4800 <= wweia_code < 4900:
                    food_groups['cereals'] += 1
                    
            elif 5000 <= wweia_code < 6000:
                food_groups['snacks_sweets'] += 1
                # Subcategories
                if 5000 <= wweia_code < 5100:
                    food_groups['savory_snacks'] += 1
                elif 5200 <= wweia_code < 5300:
                    food_groups['crackers'] += 1
                elif 5400 <= wweia_code < 5500:
                    food_groups['snack_bars'] += 1
                elif 5500 <= wweia_code < 5600:
                    food_groups['sweet_bakery'] += 1
                elif 5700 <= wweia_code < 5800:
                    food_groups['candy'] += 1
                elif 5800 <= wweia_code < 5900:
                    food_groups['desserts'] += 1
                    
            elif 6000 <= wweia_code < 6400:
                food_groups['fruits'] += 1
                # Subcategories
                if 7000 <= wweia_code < 7100:
                    food_groups['juice'] += 1
                    
            elif 6400 <= wweia_code < 7000:
                food_groups['vegetables'] += 1
                
            elif 7000 <= wweia_code < 8000:
                food_groups['beverages'] += 1
                # Subcategories
                if 7100 <= wweia_code < 7200:
                    food_groups['diet_beverages'] += 1
                elif 7200 <= wweia_code < 7300:
                    food_groups['sweetened_beverages'] += 1
                elif 7300 <= wweia_code < 7400:
                    food_groups['coffee_tea'] += 1
                elif 7700 <= wweia_code < 7800:
                    food_groups['plain_water'] += 1
                elif 7800 <= wweia_code < 7900:
                    food_groups['flavored_water'] += 1
                    
            elif 7500 <= wweia_code < 7600:
                food_groups['alcoholic_beverages'] += 1
                
            elif 8000 <= wweia_code < 8100:
                food_groups['fats_oils'] += 1
                
            elif 8400 <= wweia_code < 8500:
                food_groups['condiments_sauces'] += 1
                
            elif 8800 <= wweia_code < 8900:
                food_groups['sugars'] += 1
                
            elif 9000 <= wweia_code < 9700:
                food_groups['baby_foods'] += 1
                # Subcategories
                if 9000 <= wweia_code < 9100:
                    food_groups['baby_foods'] += 1
                elif 9200 <= wweia_code < 9300:
                    food_groups['baby_beverages'] += 1
                elif 9400 <= wweia_code < 9500:
                    food_groups['infant_formulas'] += 1
                elif 9600 <= wweia_code < 9700:
                    food_groups['human_milk'] += 1
                    
            elif 9800 <= wweia_code < 10000:
                food_groups['other'] += 1
                
            else:
                food_groups['other'] += 1
                
        return food_groups
    
    def _bin_nutritional_value(self, value, nutrient_type):
        """
        Bin nutritional values into categorical levels for better clustering.
        This helps create more distinct clusters by grouping similar nutritional profiles.
        """
        if nutrient_type == 'protein':
            if value < 10: return 0      # Low protein
            elif value < 25: return 1    # Medium protein
            elif value < 50: return 2    # High protein
            else: return 3               # Very high protein
        elif nutrient_type == 'carb':
            if value < 20: return 0      # Low carb
            elif value < 50: return 1    # Medium carb
            elif value < 100: return 2   # High carb
            else: return 3               # Very high carb
        elif nutrient_type == 'fat':
            if value < 10: return 0      # Low fat
            elif value < 25: return 1    # Medium fat
            elif value < 50: return 2    # High fat
            else: return 3               # Very high fat
        elif nutrient_type == 'energy':
            if value < 200: return 0     # Low calorie
            elif value < 500: return 1   # Medium calorie
            elif value < 800: return 2   # High calorie
            else: return 3               # Very high calorie
        else:
            return 0  # Default
    
    
    def _calculate_nutritional_balance(self, meal_nutrition):
        """
        Calculate a nutritional balance score (0-1).
        Higher score means more balanced nutrition.
        """
        score = 0
        
        # Check macronutrient balance
        protein = meal_nutrition.get('Protein (g)', 0)
        carbs = meal_nutrition.get('Carbohydrate (g)', 0)
        fat = meal_nutrition.get('Total Fat (g)', 0)
        energy = meal_nutrition.get('Energy (kcal)', 0)
        
        if energy > 0:
            # Ideal ratios: Protein 15-25%, Carbs 45-65%, Fat 20-35%
            protein_ratio = protein * 4 / energy
            carb_ratio = carbs * 4 / energy
            fat_ratio = fat * 9 / energy
            
            # Score based on how close to ideal ratios
            if 0.15 <= protein_ratio <= 0.25:
                score += 0.3
            if 0.45 <= carb_ratio <= 0.65:
                score += 0.3
            if 0.20 <= fat_ratio <= 0.35:
                score += 0.3
            
            # Bonus for fiber
            fiber = meal_nutrition.get('Fiber (total dietary) (g)', 0)
            if fiber >= 3:  # At least 3g fiber
                score += 0.1
        
        return min(score, 1.0)
    


    def prepare_meal_features(self, normalize_method='zscore', feature_type='hybrid', use_cache=True):
        """
        Create feature matrix for clustering meals by meal type.
        
        Args:
            normalize_method: part of the cache key only; features are always z-scored
                (StandardScaler), also for 'minmax'
            feature_type: 'nutritional' or 'hybrid' (the pipeline uses 'hybrid')
            use_cache: Whether to use cached features if available (a cache older
                than 24 hours is recomputed)
            
        Returns:
            Dictionary with feature matrices for each meal type
        """
        print("Preparing meal features for clustering by meal type...")
        print(f"Feature type: {feature_type}")
        
        # Check if we can load cached features
        if use_cache and self._has_cached_features(feature_type, normalize_method):
            print("Loading cached feature matrix...")
            if self._load_cached_features(feature_type, normalize_method):
                print("Successfully loaded cached features!")
                return self.meal_type_features
            else:
                print("Failed to load cached features, computing from scratch...")
        
        # Compute features from scratch
        for meal_type in self.meal_types:
            print(f"Processing {meal_type} meals...")
            meal_df = self.meal_data[meal_type]
            
            all_features = []
            meal_mapping = {}
            
            for meal_id, meal in tqdm(meal_df.iterrows(), total=len(meal_df)):
                try:
                    # Calculate nutritional metrics
                    meal_nutrition = calculate_meal_nutrition_from_df(meal, self.nutrient_data)
                    
                    # Extract features based on type
                    if feature_type == 'hybrid':
                        features = self._extract_hybrid_features(meal, meal_nutrition)
                    else:
                        features = self._extract_nutritional_features(meal, meal_nutrition)
                    
                    features['meal_id'] = meal_id
                    
                    all_features.append(features)
                    meal_mapping[len(all_features) - 1] = meal_id
                    
                except Exception as e:
                    print(f"Error processing {meal_type} meal {meal_id}: {e}")
                    continue
            
            if not all_features:
                print(f"No valid features for {meal_type}")
                continue
            
            # Convert to DataFrame
            feature_df = pd.DataFrame(all_features)
            
            # Separate metadata from features
            metadata_cols = ['meal_id']
            feature_cols = [col for col in feature_df.columns if col not in metadata_cols]
            
            # Normalize features
            if normalize_method == 'zscore':
                scaler = StandardScaler()
            else:
                scaler = StandardScaler()  # Default to zscore for better clustering
                
            feature_df[feature_cols] = scaler.fit_transform(feature_df[feature_cols])
            
            # Store results for this meal type
            self.meal_type_features[meal_type] = feature_df[feature_cols]
            self.meal_type_metadata[meal_type] = feature_df[metadata_cols]
            
            print(f"  {meal_type}: {len(feature_df)} meals with {len(feature_cols)} {feature_type} features")
        
        # Cache the computed features
        if use_cache:
            print("Caching computed features for future use...")
            self._save_cached_features(feature_type, normalize_method)
        
        return self.meal_type_features
    

    # ===================== CACHING (features only) =====================

    def _get_feature_cache_path(self, feature_type, normalize_method):
        """Get the path for cached features."""
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(
            self.output_dir, f"cached_features_{feature_type}_{normalize_method}.pkl"
        )
    
    def _has_cached_features(self, feature_type, normalize_method):
        """Check if cached features exist and are valid."""
        cache_path = self._get_feature_cache_path(feature_type, normalize_method)
        
        if not os.path.exists(cache_path):
            return False
        
        # Check if cache is recent (less than 24 hours old)
        cache_age = time.time() - os.path.getmtime(cache_path)
        if cache_age > 24 * 3600:  # 24 hours
            print(f"Cache is {cache_age/3600:.1f} hours old, will recompute")
            return False
        
        return True
    
    def _load_cached_features(self, feature_type, normalize_method):
        """Load cached features from disk."""
        try:
            cache_path = self._get_feature_cache_path(feature_type, normalize_method)
            cached_data = pd.read_pickle(cache_path)
            
            # Validate cached data structure
            required_keys = ['meal_type_features', 'meal_type_metadata', 'feature_type', 'normalize_method', 'timestamp']
            if not all(key in cached_data for key in required_keys):
                print("Cached data missing required keys")
                return False
            
            # Check if feature type matches
            if cached_data['feature_type'] != feature_type:
                print(f"Feature type mismatch: cached={cached_data['feature_type']}, requested={feature_type}")
                return False
            
            # Check if normalization method matches
            if cached_data['normalize_method'] != normalize_method:
                print(f"Normalization method mismatch: cached={cached_data['normalize_method']}, requested={normalize_method}")
                return False
            
            # Load the features
            self.meal_type_features = cached_data['meal_type_features']
            self.meal_type_metadata = cached_data['meal_type_metadata']
            
            print(f"Loaded cached features from {cache_path}")
            print(f"Cache timestamp: {cached_data['timestamp']}")
            
            return True
            
        except Exception as e:
            print(f"Error loading cached features: {e}")
            return False
    
    def _save_cached_features(self, feature_type, normalize_method):
        """Save computed features to cache."""
        try:
            cache_path = self._get_feature_cache_path(feature_type, normalize_method)
            
            cached_data = {
                'meal_type_features': self.meal_type_features,
                'meal_type_metadata': self.meal_type_metadata,
                'feature_type': feature_type,
                'normalize_method': normalize_method,
                'timestamp': time.time(),
                'meal_counts': {mt: len(features) for mt, features in self.meal_type_features.items()}
            }
            
            pd.to_pickle(cached_data, cache_path)
            print(f"Features cached to {cache_path}")
            
            # Print cache summary
            print("Cached feature summary:")
            for meal_type, count in cached_data['meal_counts'].items():
                if meal_type in self.meal_type_features:
                    feature_count = len(self.meal_type_features[meal_type].columns)
                    print(f"  {meal_type}: {count} meals, {feature_count} features")
            
            return True
            
        except Exception as e:
            print(f"Error saving cached features: {e}")
            return False

    # ===================== CLUSTERING =====================

    def _cluster_meal_type_enhanced_hdbscan(self, meal_type, min_cluster_size=100, min_samples=50):
        """
        HDBSCAN on the full feature matrix (_cluster_meal_type_hdbscan), then merge any cluster
        smaller than min_cluster_size into a large one. The low-variance/sparse-filtered
        features (_select_clustering_features) are used only for that merge and are stored as
        meal_type_clusters[meal_type]['features'], which the statistics and the t-SNE figures use.
        """
        features = self.meal_type_features[meal_type]
        
        print(f"  Enhanced HDBSCAN clustering for {meal_type}...")
        print(f"  Initial features: {len(features.columns)}")
        
        # Step 1: Feature Selection - Remove zero/low variance features
        print("  Step 1: Feature selection...")
        selected_features = self._select_clustering_features(features)
        print(f"  Selected features: {len(selected_features.columns)}")
        
        # Step 2: Perform HDBSCAN clustering with meal-type-specific parameters
        print("  Step 2: HDBSCAN clustering...")
        labels = self._cluster_meal_type_hdbscan(meal_type, min_cluster_size, min_samples)
        
        # Step 3: Post-process clusters - merge small clusters
        print("  Step 3: Post-processing clusters...")
        final_labels = self._merge_small_clusters(selected_features, labels, min_cluster_size)
        
        # Update stored results
        self.meal_type_clusters[meal_type]['labels'] = final_labels
        self.meal_type_clusters[meal_type]['features'] = selected_features
        
        return final_labels
    
    def _select_clustering_features(self, features):
        """
        Select features for clustering by removing zero/low variance features.
        """
        # Calculate feature variance
        feature_variance = features.var()
        
        # Remove features with zero or very low variance
        min_variance = 0.001  # Minimum variance threshold
        selected_features = features.loc[:, feature_variance > min_variance]
        
        # Also remove features that are mostly zeros (sparse features)
        zero_ratio = (features == 0).mean()
        selected_features = selected_features.loc[:, zero_ratio < 0.95]  # Keep features with <95% zeros
        
        print(f"    Removed {len(features.columns) - len(selected_features.columns)} low-variance features")
        
        return selected_features
    
    def _merge_small_clusters(self, features, labels, min_cluster_size):
        """
        Merge small clusters into larger, meaningful clusters.
        """
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Exclude noise points
        
        # Find small clusters
        small_clusters = []
        large_clusters = []
        
        for label in unique_labels:
            cluster_size = (labels == label).sum()
            if cluster_size < min_cluster_size:
                small_clusters.append(label)
            else:
                large_clusters.append(label)
        
        if not small_clusters:
            print("    All clusters meet minimum size requirement")
            return labels
        
        print(f"    Found {len(small_clusters)} small clusters to merge")
        
        # Create new labels
        new_labels = labels.copy()
        
        # Strategy 1: Try to merge small clusters with similar large clusters
        for small_label in small_clusters:
            small_cluster_data = features[labels == small_label]
            small_centroid = small_cluster_data.mean()
            
            best_match = None
            best_similarity = -1
            
            # Find most similar large cluster
            for large_label in large_clusters:
                large_cluster_data = features[labels == large_label]
                large_centroid = large_cluster_data.mean()
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(small_centroid, large_centroid)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = large_label
            
            # Merge if similarity is high enough
            if best_similarity > 0.7:  # High similarity threshold
                new_labels[labels == small_label] = best_match
                print(f"      Merged cluster {small_label} (size: {(labels == small_label).sum()}) into cluster {best_match}")
            else:
                # If no good match, assign to nearest large cluster by distance
                nearest_cluster = self._find_nearest_cluster(small_centroid, features, labels, large_clusters)
                new_labels[labels == small_label] = nearest_cluster
                print(f"      Assigned cluster {small_label} (size: {(labels == small_label).sum()}) to nearest cluster {nearest_cluster}")
        
        # Report final cluster sizes
        final_unique_labels = set(new_labels)
        if -1 in final_unique_labels:
            final_unique_labels.remove(-1)
        
        print(f"    Final clusters: {len(final_unique_labels)}")
        for label in sorted(final_unique_labels):
            size = (new_labels == label).sum()
            print(f"      Cluster {label}: {size} meals")
        
        return new_labels
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def _find_nearest_cluster(self, centroid, features, labels, large_clusters):
        """Find the nearest large cluster by Euclidean distance."""
        best_cluster = large_clusters[0]
        best_distance = float('inf')
        
        for label in large_clusters:
            cluster_data = features[labels == label]
            cluster_centroid = cluster_data.mean()
            
            distance = np.linalg.norm(centroid - cluster_centroid)
            if distance < best_distance:
                best_distance = distance
                best_cluster = label
        
        return best_cluster
    
    def _cluster_meal_type_hdbscan(self, meal_type, min_cluster_size=50, min_samples=25):
        """
        Perform HDBSCAN clustering for a specific meal type with meal-type-specific parameters.
        """
        features = self.meal_type_features[meal_type]
        
        print(f"  HDBSCAN clustering {len(features)} meals...")
        
        # Meal-type-specific clustering parameters based on data characteristics
        if meal_type == 'breakfast':
            # Breakfast: diverse patterns, use moderate parameters
            cluster_params = {
                'min_cluster_size': max(50, min_cluster_size),
                'min_samples': max(25, min_samples),
                'metric': 'euclidean',
                'cluster_selection_method': 'eom',
                'alpha': 1.0,  # Standard alpha for diverse patterns
                'cluster_selection_epsilon': 0.1
            }
            print("  Using moderate parameters for diverse breakfast patterns")
        elif meal_type == 'lunch':
            # Lunch: moderate homogeneity, use balanced parameters
            cluster_params = {
                'min_cluster_size': max(40, min_cluster_size),
                'min_samples': max(20, min_samples),
                'metric': 'euclidean',
                'cluster_selection_method': 'eom',
                'alpha': 1.0,
                'cluster_selection_epsilon': 0.08
            }
            print("  Using balanced parameters for moderate lunch patterns")
        elif meal_type == 'dinner':
            # Dinner: more homogeneous, use tighter parameters
            cluster_params = {
                'min_cluster_size': max(35, min_cluster_size),
                'min_samples': max(18, min_samples),
                'metric': 'euclidean',
                'cluster_selection_method': 'eom',
                'alpha': 1.0,
                'cluster_selection_epsilon': 0.06
            }
            print("  Using tighter parameters for homogeneous dinner patterns")
        else:
            cluster_params = {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom',
                'alpha': 1.0,
                'cluster_selection_epsilon': 0.1
            }
        
        print(f"  Parameters: {cluster_params}")
        
        # Import HDBSCAN
        import hdbscan
        
        clusterer = hdbscan.HDBSCAN(**cluster_params)
        
        # Perform clustering
        labels = clusterer.fit_predict(features)
        
        # Store results
        self.meal_type_clusters[meal_type] = {
            'labels': labels,
            'clusterer': clusterer,
            'features': features,
            'algorithm': 'HDBSCAN'
        }
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        coverage = (len(labels) - n_noise) / len(labels) * 100
        
        print(f"  {meal_type}: {n_clusters} clusters, {n_noise} noise points")
        print(f"  Coverage: {coverage:.1f}% of meals in clusters")
        
        # Report cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label != -1:
                print(f"    Cluster {label}: {count} meals")
        
        # Report noise percentage
        if n_noise > 0:
            print(f"    Noise: {n_noise} meals ({n_noise/len(labels)*100:.1f}%)")
        
        # Save feature matrices and generate diagnosis report
        print("  Saving feature matrices and generating diagnosis report...")
        self._save_feature_matrices(meal_type, features)
        
        # Create clusterer info for diagnosis report
        clusterer_info = {
            'algorithm': 'HDBSCAN',
            'parameters': cluster_params
        }
        
        # Generate comprehensive diagnosis report
        self._write_cluster_diagnosis_report(meal_type, features, labels, clusterer_info)
        
        return labels
    

    def _cluster_meal_type_kmeans(self, meal_type, n_clusters=5):
        """KMeans on the full feature matrix of one meal type (`--method kmeans`)."""
        from sklearn.cluster import KMeans
        
        features = self.meal_type_features[meal_type]
        
        print(f"  Clustering {len(features)} meals with {len(features.columns)} nutritional features...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Store results
        self.meal_type_clusters[meal_type] = {
            'labels': labels,
            'clusterer': kmeans,
            'features': features
        }
        
        print(f"  {meal_type}: {n_clusters} clusters")
        
        # Save feature matrices and generate diagnosis report
        print("  Saving feature matrices and generating diagnosis report...")
        self._save_feature_matrices(meal_type, features)
        
        # Create clusterer info for diagnosis report
        clusterer_info = {
            'algorithm': 'KMeans',
            'parameters': {'n_clusters': n_clusters, 'random_state': 42}
        }
        
        # Generate comprehensive diagnosis report
        self._write_cluster_diagnosis_report(meal_type, features, labels, clusterer_info)


    def perform_clustering_by_meal_type(self, method='enhanced_hdbscan', **kwargs):
        """
        Cluster each meal type separately on its full feature matrix.
        
        Args:
            method: 'enhanced_hdbscan' (the pipeline), 'hdbscan', 'hierarchical' or 'kmeans'
            **kwargs: passed to the method (run_clustering passes min_cluster_size and min_samples)
        """
        print(f"Performing clustering by meal type using {method}...")
        print("NOTE: Clustering is performed on FULL 20D nutritional data")
        
        for meal_type in self.meal_types:
            if meal_type not in self.meal_type_features:
                print(f"Skipping {meal_type} - no features available")
                continue
                
            print(f"\nClustering {meal_type} meals on {len(self.meal_type_features[meal_type].columns)}D nutritional data...")
            
            if method == 'enhanced_hdbscan':
                self._cluster_meal_type_enhanced_hdbscan(meal_type, **kwargs)
            elif method == 'hierarchical':
                self._cluster_meal_type_hierarchical(meal_type, **kwargs)
            elif method == 'hdbscan':
                self._cluster_meal_type_hdbscan(meal_type, **kwargs)
            elif method == 'kmeans':
                self._cluster_meal_type_kmeans(meal_type, **kwargs)
    
    def _cluster_meal_type_hierarchical(self, meal_type, n_clusters=12, min_cluster_size=200):
        """
        Ward agglomerative clustering of one meal type on its (re-standardised) full feature
        matrix (`--method hierarchical`). min_cluster_size is only reported, not enforced.
        """
        features = self.meal_type_features[meal_type]
        
        print(f"  Clustering {len(features)} meals with {len(features.columns)} hybrid features...")
        
        # Meal-type-specific clustering parameters
        if meal_type == 'breakfast':
            # Breakfast: diverse patterns, aim for 10-15 clusters
            target_clusters = min(n_clusters, 15)
            print("  Using hierarchical clustering for diverse breakfast patterns")
        elif meal_type == 'lunch':
            # Lunch: homogeneous patterns, aim for 8-12 clusters
            target_clusters = min(n_clusters, 12)
            print("  Using hierarchical clustering for homogeneous lunch patterns")
        elif meal_type == 'dinner':
            # Dinner: balanced patterns, aim for 10-14 clusters
            target_clusters = min(n_clusters, 14)
            print("  Using hierarchical clustering for balanced dinner patterns")
        else:
            target_clusters = n_clusters
        
        print(f"  Target clusters: {target_clusters}")
        print(f"  Minimum cluster size: {min_cluster_size}")
        
        # Use AgglomerativeClustering for hierarchical clustering
        from sklearn.cluster import AgglomerativeClustering
        
        # Scale features for better clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform hierarchical clustering
        clusterer = AgglomerativeClustering(
            n_clusters=target_clusters,
            linkage='ward',  # Ward linkage for balanced cluster sizes
            metric='euclidean'
        )
        
        # CLUSTER ON FULL DIMENSIONAL DATA
        labels = clusterer.fit_predict(features_scaled)
        
        # Store results
        self.meal_type_clusters[meal_type] = {
            'labels': labels,
            'clusterer': clusterer,
            'features': features,
            'algorithm': 'AgglomerativeClustering'
        }
        
        n_clusters_found = len(set(labels))
        coverage = 100.0  # Hierarchical clustering assigns all points to clusters
        
        print(f"  {meal_type}: {n_clusters_found} clusters, 0 noise points")
        print(f"  Coverage: {coverage:.1f}% of meals in clusters")
        
        # Report cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = []
        for label, count in zip(unique_labels, counts):
            cluster_sizes.append(count)
            print(f"    Cluster {label}: {count} meals")
        
        # Calculate cluster size statistics
        avg_size = np.mean(cluster_sizes)
        min_size = np.min(cluster_sizes)
        max_size = np.max(cluster_sizes)
        
        print(f"  Cluster size stats: min={min_size:.0f}, avg={avg_size:.0f}, max={max_size:.0f}")
        
        # Check if clusters meet size requirements
        small_clusters = [size for size in cluster_sizes if size < min_cluster_size]
        if small_clusters:
            print(f"  ⚠️  {len(small_clusters)} clusters below minimum size ({min_cluster_size})")
            print(f"     Smallest cluster: {min_size} meals")
        else:
            print("  ✅ All clusters meet minimum size requirement")
        
        # Save feature matrices and generate diagnosis report
        print("  Saving feature matrices and generating diagnosis report...")
        self._save_feature_matrices(meal_type, features)
        
        # Create clusterer info for diagnosis report
        clusterer_info = {
            'algorithm': 'AgglomerativeClustering',
            'parameters': {
                'n_clusters': target_clusters,
                'linkage': 'ward',
                'metric': 'euclidean'
            }
        }
        
        # Generate comprehensive diagnosis report
        self._write_cluster_diagnosis_report(meal_type, features, labels, clusterer_info)
        

        return labels


    # ===================== VALIDATION / REPORTING (non-plot) =====================

    def validate_clusters_statistically(self, alpha=0.01, delta_min=0.15):
        """
        Validate clusters using statistical tests on nutritional differences.
        Implements the two-part hurdle test from FoodAtlas methodology.
        
        Args:
            alpha: Significance level for FDR correction
            delta_min: Minimum absolute difference for significance
        """
        print("\nValidating clusters using statistical tests...")
        
        for meal_type in self.meal_types:
            if meal_type not in self.meal_type_clusters:
                continue
                
            print(f"\nValidating {meal_type} clusters...")
            
            clusters = self.meal_type_clusters[meal_type]
            labels = clusters['labels']
            features = clusters['features']
            
            # Get non-noise clusters
            unique_labels = [l for l in np.unique(labels) if l != -1]
            
            if len(unique_labels) < 2:
                print(f"  {meal_type}: Need at least 2 clusters for validation")
                continue
            
            # Calculate statistics for each cluster
            cluster_stats = {}
            
            for cluster_id in unique_labels:
                cluster_mask = labels == cluster_id
                cluster_data = features[cluster_mask]
                other_data = features[~cluster_mask]
                
                cluster_stats[cluster_id] = {}
                
                # Test each nutritional feature
                for feature in features.columns:
                    stats_result = self._two_part_hurdle_test(
                        cluster_data[feature], 
                        other_data[feature]
                    )
                    cluster_stats[cluster_id][feature] = stats_result
            
            self.meal_type_stats[meal_type] = cluster_stats
            
            # Apply FDR correction and filter significant features
            significant_features = self._apply_fdr_correction(
                meal_type, cluster_stats, alpha, delta_min
            )
            
            if significant_features:
                self.significant_clusters[meal_type] = significant_features
                print(f"  {meal_type}: {len(significant_features)} significant clusters")
            else:
                print(f"  {meal_type}: No significant clusters found")
    
    def _two_part_hurdle_test(self, cluster_data, other_data):
        """
        Two-part hurdle test: prevalence (Fisher exact) + intensity (Mann-Whitney).
        Adapted from FoodAtlas methodology.
        """
        # Prevalence test (non-zero vs zero)
        cluster_prev = (cluster_data != 0).mean()
        other_prev = (other_data != 0).mean()
        
        # Create contingency table
        cluster_nonzero = (cluster_data != 0).sum()
        cluster_zero = (cluster_data == 0).sum()
        other_nonzero = (other_data != 0).sum()
        other_zero = (other_data == 0).sum()
        
        try:
            if min(cluster_nonzero, cluster_zero, other_nonzero, other_zero) > 0:
                contingency = np.array([[cluster_nonzero, cluster_zero],
                                       [other_nonzero, other_zero]])
                _, p_prev = stats.fisher_exact(contingency)
            else:
                p_prev = 1.0
        except:
            p_prev = 1.0
        
        # Intensity test (Mann-Whitney on non-zero values)
        cluster_nonzero_data = cluster_data[cluster_data != 0]
        other_nonzero_data = other_data[other_data != 0]
        
        if len(cluster_nonzero_data) > 0 and len(other_nonzero_data) > 0:
            try:
                _, p_int = stats.mannwhitneyu(
                    cluster_nonzero_data, other_nonzero_data, 
                    alternative='two-sided'
                )
            except:
                p_int = 1.0
        else:
            p_int = 1.0
        
        # Effect size (Cohen's d)
        if len(cluster_data) > 0 and len(other_data) > 0:
            try:
                pooled_std = np.sqrt(((len(cluster_data) - 1) * cluster_data.var() + 
                                     (len(other_data) - 1) * other_data.var()) / 
                                    (len(cluster_data) + len(other_data) - 2))
                if pooled_std > 0:
                    cohen_d = (cluster_data.mean() - other_data.mean()) / pooled_std
                else:
                    cohen_d = 0
            except:
                cohen_d = 0
        else:
            cohen_d = 0
        
        # Absolute difference
        abs_delta = abs(cluster_data.mean() - other_data.mean())
        
        return {
            'p_prev': p_prev,
            'p_int': p_int,
            'cohen_d': cohen_d,
            'abs_delta': abs_delta,
            'cluster_mean': cluster_data.mean(),
            'other_mean': other_data.mean(),
            'cluster_prev': cluster_prev,
            'other_prev': other_prev
        }
    
    def _apply_fdr_correction(self, meal_type, cluster_stats, alpha, delta_min):
        """
        Apply FDR correction and filter significant features.
        """
        # Collect all p-values for FDR correction
        all_p_values = []
        cluster_feature_pairs = []
        
        for cluster_id, features in cluster_stats.items():
            for feature, stats_result in features.items():
                # Use the smaller (less conservative) of the prevalence and intensity p-values
                p_value = min(stats_result['p_prev'], stats_result['p_int'])
                all_p_values.append(p_value)
                cluster_feature_pairs.append((cluster_id, feature))
        
        if not all_p_values:
            return {}
        
        # Apply FDR correction
        rejected, pvals_corrected, _, _ = multipletests(
            all_p_values, method='fdr_bh', alpha=alpha
        )
        
        # Filter significant features
        significant_features = {}
        
        for cluster_id, features in cluster_stats.items():
            significant_features[cluster_id] = {}
            
            for feature, stats_result in features.items():
                pair_idx = cluster_feature_pairs.index((cluster_id, feature))
                
                # Check if feature is significant after FDR correction
                if (pvals_corrected[pair_idx] <= alpha and 
                    stats_result['abs_delta'] >= delta_min):
                    significant_features[cluster_id][feature] = {
                        **stats_result,
                        'q_value': pvals_corrected[pair_idx]
                    }
        
        return significant_features
    
    def characterize_clusters(self):
        """Create human-readable descriptions of each cluster."""
        print("\nCharacterizing significant clusters...")
        
        self.cluster_profiles = {}
        
        for meal_type in self.meal_types:
            if meal_type not in self.significant_clusters:
                continue
                
            print(f"\n{meal_type} clusters:")
            
            # Calculate total meals for this meal type
            total_meals = len(self.meal_type_clusters[meal_type]['labels'])
            
            for cluster_id, features in self.significant_clusters[meal_type].items():
                # Get cluster size
                cluster_mask = self.meal_type_clusters[meal_type]['labels'] == cluster_id
                cluster_size = cluster_mask.sum()
                cluster_percentage = self._calculate_cluster_percentage(cluster_size, total_meals)
                
                # Generate meaningful cluster name
                label = self._generate_meaningful_cluster_name(meal_type, cluster_id, features, features)
                
                # Identify most distinctive features
                distinctive_features = []
                for feature, stats in features.items():
                    if stats['abs_delta'] >= 0.2:  # Higher threshold for distinctive
                        direction = "high" if stats['cluster_mean'] > stats['other_mean'] else "low"
                        distinctive_features.append(f"{direction} {feature}")
                
                # Create description
                if distinctive_features:
                    description = f"{meal_type.title()} meals with " + ", ".join(distinctive_features)
                else:
                    description = f"General {meal_type} meal pattern"
                
                self.cluster_profiles[f"{meal_type}_{cluster_id}"] = {
                    'meal_type': meal_type,
                    'cluster_id': cluster_id,
                    'label': label,
                    'description': description,
                    'size': cluster_size,
                    'percentage': cluster_percentage,
                    'distinctive_features': distinctive_features,
                    'nutritional_profile': {
                        feature: {
                            'cluster_mean': stats['cluster_mean'],
                            'other_mean': stats['other_mean'],
                            'difference': stats['cluster_mean'] - stats['other_mean'],
                            'effect_size': stats['cohen_d'],
                            'significance': stats['q_value']
                        }
                        for feature, stats in features.items()
                    }
                }
                
                print(f"  {label}: {cluster_size} meals ({cluster_percentage:.1f}%)")
                print(f"    {description}")
                if distinctive_features:
                    print(f"    Distinctive: {', '.join(distinctive_features[:3])}")
        
        return self.cluster_profiles

    def _generate_meaningful_cluster_name(self, meal_type, cluster_id, features, stats):
        """
        Name a cluster after its (up to) three significant features with the largest
        abs_delta >= 0.2, e.g. "Breakfast - high protein & low carbs". Used in the log and
        feature_contribution_report.csv; the figures prefer config/cluster_names.json.
        `stats` is not used.
        """
        # Get the most distinctive features (largest abs_delta)
        distinctive_features = []
        for feature, stat in features.items():
            if stat['abs_delta'] >= 0.2:  # Only consider truly distinctive features
                direction = "high" if stat['cluster_mean'] > stat['other_mean'] else "low"
                distinctive_features.append((feature, direction, stat['abs_delta']))
        
        # Sort by abs_delta (most distinctive first)
        distinctive_features.sort(key=lambda x: x[2], reverse=True)
        
        # Take top 3 most distinctive features
        top_features = distinctive_features[:3]
        
        if not top_features:
            return f"{meal_type.title()} Cluster {cluster_id}"
        
        # Create descriptive name
        feature_descriptions = []
        for feature, direction, _ in top_features:
            # Convert feature names to readable descriptions
            readable_name = self._get_readable_feature_name(feature)
            feature_descriptions.append(f"{direction} {readable_name}")
        
        # Combine into a meaningful name
        if len(feature_descriptions) == 1:
            name = f"{meal_type.title()} - {feature_descriptions[0]}"
        elif len(feature_descriptions) == 2:
            name = f"{meal_type.title()} - {feature_descriptions[0]} & {feature_descriptions[1]}"
        else:
            name = f"{meal_type.title()} - {feature_descriptions[0]}, {feature_descriptions[1]} & {feature_descriptions[2]}"
        
        return name
    
    def _get_readable_feature_name(self, feature):
        """Convert technical feature names to human-readable descriptions."""
        feature_map = {
            # Nutritional features
            'protein_g': 'protein',
            'carbohydrate_g': 'carbs', 
            'fat_g': 'fat',
            'fiber_g': 'fiber',
            'energy_kcal': 'calories',
            'protein_ratio': 'protein ratio',
            'carb_ratio': 'carb ratio',
            'fat_ratio': 'fat ratio',
            
            # Enhanced composition features
            'food_category_diversity': 'food diversity',
            'ingredient_count': 'ingredients',
            'portion_variability': 'portion variety',
            'nutritional_balance': 'nutritional balance',
            'calorie_density': 'calorie density'
        }
        
        return feature_map.get(feature, feature)
    

    def _calculate_cluster_percentage(self, cluster_size, total_meals):
        """Calculate what percentage of total meals this cluster represents."""
        return (cluster_size / total_meals) * 100 if total_meals > 0 else 0 

    def generate_feature_contribution_report(self, save_path=None):
        """
        Generate a comprehensive feature contribution report for each cluster.
        Similar to FoodAtlas, this shows ranked features by importance and contribution.
        """
        print("\n" + "="*80)
        print("FEATURE CONTRIBUTION REPORT FOR EACH CLUSTER")
        print("="*80)
        
        if save_path is None:
            save_path = self.output_dir
        os.makedirs(save_path, exist_ok=True)
        
        # Create detailed report for each meal type
        for meal_type in self.meal_types:
            if meal_type not in self.significant_clusters:
                continue
                
            print(f"\n{meal_type.upper()} CLUSTERS FEATURE ANALYSIS:")
            print("-" * 60)
            
            # Get total meals for this meal type
            total_meals = len(self.meal_type_clusters[meal_type]['labels'])
            
            # Analyze each cluster
            for cluster_id, features in self.significant_clusters[meal_type].items():
                self._analyze_cluster_features(meal_type, cluster_id, features, total_meals)
        
        # Save detailed report to CSV
        self._save_feature_contribution_csv(save_path)
        
        print(f"\nFeature contribution report saved to: {save_path}")
        return True
    
    def _analyze_cluster_features(self, meal_type, cluster_id, features, total_meals):
        """Analyze and display feature contributions for a specific cluster."""
        # Get cluster size and percentage
        cluster_mask = self.meal_type_clusters[meal_type]['labels'] == cluster_id
        cluster_size = cluster_mask.sum()
        cluster_percentage = self._calculate_cluster_percentage(cluster_size, total_meals)
        
        # Get cluster name
        cluster_name = self._generate_meaningful_cluster_name(meal_type, cluster_id, features, features)
        
        print(f"\n{cluster_name}")
        print(f"Cluster ID: {cluster_id} | Size: {cluster_size} meals ({cluster_percentage:.1f}%)")
        print("-" * 80)
        
        # Rank features by importance (effect size × statistical significance)
        feature_rankings = []
        for feature, stats in features.items():
            # Calculate feature importance score
            effect_size = abs(stats['cohen_d'])
            significance = -np.log10(stats['q_value']) if stats['q_value'] > 0 else 10
            importance_score = effect_size * significance
            
            feature_rankings.append({
                'feature': feature,
                'importance_score': importance_score,
                'effect_size': stats['cohen_d'],
                'significance': stats['q_value'],
                'cluster_mean': stats['cluster_mean'],
                'other_mean': stats['other_mean'],
                'difference': stats['cluster_mean'] - stats['other_mean'],
                'abs_delta': stats['abs_delta'],
                'direction': 'HIGH' if stats['cluster_mean'] > stats['other_mean'] else 'LOW'
            })
        
        # Sort by importance score (highest first)
        feature_rankings.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Display top features
        print("TOP FEATURES BY IMPORTANCE:")
        print(f"{'Rank':<4} {'Feature':<25} {'Direction':<6} {'Effect':<8} {'Significance':<12} {'Cluster':<10} {'Others':<10} {'Diff':<10}")
        print("-" * 80)
        
        for rank, feat in enumerate(feature_rankings[:10], 1):  # Top 10 features
            print(f"{rank:<4} {feat['feature']:<25} {feat['direction']:<6} "
                  f"{feat['effect_size']:<8.3f} {feat['significance']:<12.3e} "
                  f"{feat['cluster_mean']:<10.3f} {feat['other_mean']:<10.3f} "
                  f"{feat['difference']:<10.3f}")
        
        # Show feature interpretation
        print("\nCLUSTER CHARACTERISTICS:")
        top_features = feature_rankings  # Top most important features
        for feat in top_features:
            readable_name = self._get_readable_feature_name(feat['feature'])
            direction = "high" if feat['direction'] == 'HIGH' else "low"
            print(f"  • {direction.upper()} {readable_name}: {feat['cluster_mean']:.3f} vs {feat['other_mean']:.3f} "
                  f"(difference: {feat['difference']:+.3f}, effect: {feat['effect_size']:.3f})")
        
        # Show nutritional summary if available
        self._show_nutritional_summary(meal_type, cluster_id, features)
    
    def _show_nutritional_summary(self, meal_type, cluster_id, features):
        """Show a nutritional summary for the cluster."""
        print("\nNUTRITIONAL PROFILE SUMMARY:")
        
        # Key nutritional features to highlight
        key_features = ['protein_g', 'carbohydrate_g', 'fat_g', 'energy_kcal', 
                       'protein_ratio', 'carb_ratio', 'fat_ratio', 'fiber_g']
        
        for feature in key_features:
            if feature in features:
                stats = features[feature]
                direction = "HIGH" if stats['cluster_mean'] > stats['other_mean'] else "LOW"
                diff = stats['cluster_mean'] - stats['other_mean']
                
                if feature in ['protein_g', 'carbohydrate_g', 'fat_g', 'fiber_g']:
                    unit = "g"
                elif feature == 'energy_kcal':
                    unit = "kcal"
                else:
                    unit = ""
                
                print(f"  • {feature}: {direction} ({stats['cluster_mean']:.2f}{unit} vs {stats['other_mean']:.2f}{unit}, diff: {diff:+.2f}{unit})")
    

    def _save_feature_contribution_csv(self, save_path):
        """Save detailed feature contribution data to CSV for further analysis."""
        all_cluster_data = []
        
        for meal_type in self.meal_types:
            if meal_type not in self.significant_clusters:
                continue
                
            for cluster_id, features in self.significant_clusters[meal_type].items():
                # Get cluster info
                cluster_mask = self.meal_type_clusters[meal_type]['labels'] == cluster_id
                cluster_size = cluster_mask.sum()
                total_meals = len(self.meal_type_clusters[meal_type]['labels'])
                cluster_percentage = self._calculate_cluster_percentage(cluster_size, total_meals)
                cluster_name = self._generate_meaningful_cluster_name(meal_type, cluster_id, features, features)
                
                # Process each feature
                for feature, stats in features.items():
                    effect_size = abs(stats['cohen_d'])
                    significance = -np.log10(stats['q_value']) if stats['q_value'] > 0 else 10
                    importance_score = effect_size * significance
                    
                    all_cluster_data.append({
                        'meal_type': meal_type,
                        'cluster_id': cluster_id,
                        'cluster_name': cluster_name,
                        'cluster_size': cluster_size,
                        'cluster_percentage': cluster_percentage,
                        'feature': feature,
                        'feature_readable': self._get_readable_feature_name(feature),
                        'importance_score': importance_score,
                        'effect_size': stats['cohen_d'],
                        'significance': stats['q_value'],
                        'cluster_mean': stats['cluster_mean'],
                        'other_mean': stats['other_mean'],
                        'difference': stats['cluster_mean'] - stats['other_mean'],
                        'abs_delta': stats['abs_delta'],
                        'direction': 'HIGH' if stats['cluster_mean'] > stats['other_mean'] else 'LOW'
                    })
        
        # Convert to DataFrame and save
        if all_cluster_data:
            df = pd.DataFrame(all_cluster_data)
            df = df.sort_values(['meal_type', 'cluster_id', 'importance_score'], 
                               ascending=[True, True, False])
            
            base = save_path if isinstance(save_path, str) else self.output_dir
            csv_file = os.path.join(base, "feature_contribution_report.csv")
            df.to_csv(csv_file, index=False)
            print(f"\nDetailed feature contribution data saved to: {csv_file}")
            
            # Show summary statistics
            print("\nREPORT SUMMARY:")
            print(f"  Total clusters analyzed: {len(set([(d['meal_type'], d['cluster_id']) for d in all_cluster_data]))}")
            print(f"  Total feature-cluster combinations: {len(all_cluster_data)}")
            print(f"  Features per cluster (avg): {len(all_cluster_data) / len(set([(d['meal_type'], d['cluster_id']) for d in all_cluster_data])):.1f}")
    

    def diagnose_clustering_issues(self):
        """
        Log feature-distribution diagnostics per meal type and compare feature variances
        across meal types (printed only; no files are written).
        """
        print("\n" + "="*80)
        print("CLUSTERING DIAGNOSIS - INVESTIGATING MEAL TYPE DIFFERENCES")
        print("="*80)
        
        for meal_type in self.meal_types:
            if meal_type not in self.meal_type_features:
                print(f"\n{meal_type.upper()}: NO FEATURES AVAILABLE")
                continue
                
            print(f"\n{meal_type.upper()} DIAGNOSIS:")
            print("-" * 60)
            
            # Analyze feature distributions
            self._analyze_feature_distributions(meal_type)

            # Compare with other meal types
            self._compare_meal_type_characteristics(meal_type)


    def _analyze_feature_distributions(self, meal_type):
        """Analyze the distribution of features for a meal type."""
        features = self.meal_type_features[meal_type]
        
        print("\nFEATURE DISTRIBUTION ANALYSIS:")
        print(f"  Total meals: {len(features)}")
        print(f"  Feature dimensions: {len(features.columns)}")
        
        # Calculate feature statistics
        feature_stats = {}
        for col in features.columns:
            feature_stats[col] = {
                'mean': features[col].mean(),
                'std': features[col].std(),
                'min': features[col].min(),
                'max': features[col].max(),
                'variance': features[col].var(),
                'skewness': features[col].skew(),
                'kurtosis': features[col].kurtosis()
            }
        
        # Sort features by variance (most to least variable)
        sorted_features = sorted(feature_stats.items(), 
                               key=lambda x: x[1]['variance'], reverse=True)
        
        print("\nTOP 10 MOST VARIABLE FEATURES:")
        print(f"{'Feature':<25} {'Variance':<12} {'Std':<10} {'Range':<15} {'Skew':<8}")
        print("-" * 70)
        
        for feature, stats in sorted_features[:10]:
            range_val = stats['max'] - stats['min']
            print(f"{feature:<25} {stats['variance']:<12.4f} {stats['std']:<10.4f} "
                  f"{range_val:<15.4f} {stats['skewness']:<8.3f}")
        
        # Check for potential issues
        print("\nPOTENTIAL ISSUES IDENTIFIED:")
        
        # Check for low variance features (might cause clustering issues)
        low_variance_features = [f for f, s in feature_stats.items() if s['variance'] < 0.1]
        if low_variance_features:
            print(f"  • Low variance features (<0.1): {len(low_variance_features)} features")
            print(f"    These might not contribute to clustering: {low_variance_features[:5]}")
        
        # Check for highly skewed features
        highly_skewed = [f for f, s in feature_stats.items() if abs(s['skewness']) > 3]
        if highly_skewed:
            print(f"  • Highly skewed features (|skewness|>3): {len(highly_skewed)} features")
            print(f"    These might need transformation: {highly_skewed[:5]}")
        
        # Check for binary features (should be 0/1)
        binary_features = []
        for col in features.columns:
            unique_vals = features[col].nunique()
            if unique_vals == 2:
                binary_features.append(col)
        
        if binary_features:
            print(f"  • Binary features detected: {len(binary_features)} features")
            print(f"    These should be 0/1: {binary_features}")
        
        return feature_stats

    def _compare_meal_type_characteristics(self, meal_type):
        """Compare characteristics of this meal type with others."""
        if not hasattr(self, 'meal_type_features') or len(self.meal_type_features) < 2:
            return
        
        print("\nCOMPARISON WITH OTHER MEAL TYPES:")
        
        # Get feature matrices for comparison
        current_features = self.meal_type_features[meal_type]
        other_meal_types = [mt for mt in self.meal_types if mt != meal_type and mt in self.meal_type_features]
        
        if not other_meal_types:
            return
        
        print(f"  Comparing {meal_type} with: {', '.join(other_meal_types)}")
        
        # Compare feature variances
        current_variances = {col: current_features[col].var() for col in current_features.columns}
        
        for other_type in other_meal_types:
            other_features = self.meal_type_features[other_type]
            other_variances = {col: other_features[col].var() for col in other_features.columns}
            
            # Calculate correlation of variances
            common_features = set(current_variances.keys()) & set(other_variances.keys())
            if common_features:
                current_vals = [current_variances[f] for f in common_features]
                other_vals = [other_variances[f] for f in common_features]
                
                correlation = np.corrcoef(current_vals, other_vals)[0, 1]
                
                print(f"  • Variance correlation with {other_type}: {correlation:.3f}")
                
                # Check for significant differences
                significant_diff = 0
                for feature in common_features:
                    if other_variances[feature] != 0:  # Avoid division by zero
                        ratio = current_variances[feature] / other_variances[feature]
                        if ratio > 2 or ratio < 0.5:
                            significant_diff += 1
                    else:
                        # Handle zero variance case
                        if current_variances[feature] != 0:
                            significant_diff += 1  # Current has variance, other doesn't
                
                if significant_diff > 0:
                    print(f"    {significant_diff} features have >2x different variance")
        
        # Check if this meal type has unusual characteristics
        current_mean_var = np.mean(list(current_variances.values()))
        other_mean_vars = []
        
        for other_type in other_meal_types:
            if other_type in self.meal_type_features:
                other_features = self.meal_type_features[other_type]
                other_var = np.mean([other_features[col].var() for col in other_features.columns])
                other_mean_vars.append(other_var)
        
        if other_mean_vars:
            overall_mean_var = np.mean(other_mean_vars)
            if current_mean_var < overall_mean_var * 0.5:
                print(f"  ⚠️  {meal_type} has unusually LOW feature variance ({current_mean_var:.3f} vs {overall_mean_var:.3f})")
                print("     This could explain poor clustering - features are too similar")
            elif current_mean_var > overall_mean_var * 2:
                print(f"  ⚠️  {meal_type} has unusually HIGH feature variance ({current_mean_var:.3f} vs {overall_mean_var:.3f})")
                print("     This could cause clustering to be too sensitive") 

    # ===================== PERSISTENCE / UTILITIES =====================

    def _save_feature_matrices(self, meal_type, features_df, scaler=None):
        """
        Save the feature matrix to <meal>_feature_matrix_unscaled.csv (and, given a fitted
        scaler, also a scaled copy and the scaler parameters). The callers pass the
        features from prepare_meal_features, which are already z-scored, and no scaler.
        
        Args:
            meal_type (str): Type of meal (breakfast, lunch, dinner)
            features_df (pd.DataFrame): Feature matrix
            scaler: Fitted scaler object (if None, only unscaled matrix is saved)
        """
        # Save unscaled feature matrix
        unscaled_file = os.path.join(self.output_dir, f"{meal_type}_feature_matrix_unscaled.csv")
        os.makedirs(self.output_dir, exist_ok=True)
        features_df.to_csv(unscaled_file, index=False)
        print(f"  Saved unscaled feature matrix: {unscaled_file}")
        
        # Save scaled feature matrix if scaler is provided
        if scaler is not None:
            scaled_features = scaler.transform(features_df)
            scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns)
            scaled_file = os.path.join(self.output_dir, f"{meal_type}_feature_matrix_scaled.csv")
            scaled_df.to_csv(scaled_file, index=False)
            print(f"  Saved scaled feature matrix: {scaled_file}")
            
            # Save scaler parameters
            scaler_file = os.path.join(self.output_dir, f"{meal_type}_scaler_params.json")
            scaler_params = {
                'mean_': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                'scale_': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
                'var_': scaler.var_.tolist() if hasattr(scaler, 'var_') else None,
                'n_samples_seen_': scaler.n_samples_seen_ if hasattr(scaler, 'n_samples_seen_') else None
            }
            with open(scaler_file, 'w') as f:
                json.dump(scaler_params, f, indent=2)
            print(f"  Saved scaler parameters: {scaler_file}")


    def _write_cluster_diagnosis_report(self, meal_type, features_df, cluster_labels, clusterer_info, fallback_info=None):
        """
        Write comprehensive cluster diagnosis report for a meal type.
        
        Args:
            meal_type (str): Type of meal (breakfast, lunch, dinner)
            features_df (pd.DataFrame): Feature matrix
            cluster_labels (np.array): Cluster labels
            clusterer_info (dict): Information about the clustering algorithm used
            fallback_info (dict): Information about fallback clustering if used
        """
        os.makedirs(self.output_dir, exist_ok=True)
        report_file = os.path.join(self.output_dir, f"{meal_type}_cluster_diagnosis_report.txt")
        
        with open(report_file, 'w') as f:
            f.write(f"=== CLUSTER DIAGNOSIS REPORT FOR {meal_type.upper()} ===\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Basic dataset information
            f.write("=== DATASET INFORMATION ===\n")
            f.write(f"Total meals: {len(features_df)}\n")
            f.write(f"Total features: {len(features_df.columns)}\n")
            f.write(f"Feature names: {', '.join(features_df.columns)}\n\n")
            
            # Feature statistics
            f.write("=== FEATURE STATISTICS ===\n")
            f.write(f"Features with zero variance: {features_df.columns[features_df.var() == 0].tolist()}\n")
            f.write(f"Features with low variance (<0.01): {features_df.columns[features_df.var() < 0.01].tolist()}\n")
            f.write(f"Features with high variance (>100): {features_df.columns[features_df.var() > 100].tolist()}\n\n")
            
            # Clustering algorithm information
            f.write("=== CLUSTERING ALGORITHM ===\n")
            f.write(f"Algorithm: {clusterer_info.get('algorithm', 'Unknown')}\n")
            f.write(f"Parameters: {clusterer_info.get('parameters', {})}\n")
            f.write(f"Number of clusters found: {len(np.unique(cluster_labels[cluster_labels >= 0]))}\n")
            f.write(f"Number of noise points: {np.sum(cluster_labels == -1)}\n")
            f.write(f"Cluster sizes: {np.bincount(cluster_labels[cluster_labels >= 0])}\n\n")
            
            # Fallback clustering information
            if fallback_info:
                f.write("=== FALLBACK CLUSTERING USED ===\n")
                f.write(f"Reason: {fallback_info.get('reason', 'Unknown')}\n")
                f.write(f"Fallback method: {fallback_info.get('method', 'Unknown')}\n")
                f.write(f"Fallback parameters: {fallback_info.get('parameters', {})}\n\n")
            
            # Feature importance analysis
            f.write("=== FEATURE IMPORTANCE ANALYSIS ===\n")
            if len(np.unique(cluster_labels[cluster_labels >= 0])) > 1:
                # Calculate feature importance using variance between clusters
                cluster_means = []
                for cluster_id in np.unique(cluster_labels[cluster_labels >= 0]):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_means.append(features_df[cluster_mask].mean())
                
                if len(cluster_means) > 1:
                    cluster_means_df = pd.DataFrame(cluster_means)
                    feature_importance = cluster_means_df.var().sort_values(ascending=False)
                    
                    f.write("Top 20 most important features (by inter-cluster variance):\n")
                    for i, (feature, importance) in enumerate(feature_importance.head(20).items()):
                        f.write(f"  {i+1:2d}. {feature}: {importance:.6f}\n")
                    f.write("\n")
            
            # Cluster characteristics
            f.write("=== CLUSTER CHARACTERISTICS ===\n")
            for cluster_id in np.unique(cluster_labels[cluster_labels >= 0]):
                cluster_mask = cluster_labels == cluster_id
                cluster_size = np.sum(cluster_mask)
                cluster_percentage = cluster_size / len(features_df) * 100
                
                f.write(f"Cluster {cluster_id}:\n")
                f.write(f"  Size: {cluster_size} ({cluster_percentage:.1f}%)\n")
                
                # Top distinguishing features for this cluster
                cluster_mean = features_df[cluster_mask].mean()
                other_mean = features_df[~cluster_mask].mean()
                feature_differences = (cluster_mean - other_mean).abs().sort_values(ascending=False)
                
                f.write("  Top distinguishing features:\n")
                for i, (feature, diff) in enumerate(feature_differences.items()):
                    f.write(f"    {i+1}. {feature}: {cluster_mean[feature]:.3f} vs {other_mean[feature]:.3f} (diff: {diff:.3f})\n")
                f.write("\n")
        
        print(f"  Saved cluster diagnosis report: {report_file}")

    def export_cvae_training_matrices(self, meal_type: str, labels: np.ndarray):
        """
        Write the CVAE training data for one meal type: for every label (including noise, -1)
        <output_dir>/<meal>/cluster_<id>_training_data.pkl, a float32 meals x foods matrix in
        grams with the same columns for every cluster, and a .units.json manifest. Meals whose
        total weight is outside 120-3,500 g are left out.
        """
        # 1) original meals for this meal_type
        meal_df = self.meal_data[meal_type]  # loaded via load_meal_data()
        # item columns: numeric names or numeric dtype
        item_cols = [c for c in meal_df.columns
                    if (pd.api.types.is_numeric_dtype(meal_df[c]) or str(c).isdigit())]
        grams_all = (meal_df[item_cols].copy()
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0).clip(lower=0.0).astype("float32"))

        grams_all.columns = [ _canon_item_label(c) for c in grams_all.columns ]

        # if any duplicate labels collapse after canonicalization, sum them
        if len(set(grams_all.columns)) != len(grams_all.columns):
            grams_all = grams_all.groupby(level=0, axis=1).sum()

        # keep meals with a total of 120-3,500 g
        totals = grams_all.sum(axis=1)
        keep = (totals.between(120.0, 3500.0))
        grams_all = grams_all.loc[keep]
        labels = labels[keep.values]  # keep labels aligned

        # fixed column order manifest
        cols = [str(c) for c in grams_all.columns]
        out_dir = os.path.join(self.output_dir, meal_type.lower())
        os.makedirs(out_dir, exist_ok=True)

        # 2) write one file per cluster id in *grams*
        for cid in sorted(np.unique(labels)):
            mask = (labels == cid)
            df_c = grams_all.loc[mask].copy()

            # ensure cluster frame uses the same canonical labels
            df_c.columns = [ _canon_item_label(c) for c in df_c.columns ]
            if len(set(df_c.columns)) != len(df_c.columns):
                df_c = df_c.groupby(level=0, axis=1).sum()

            # align to the global column order; add any missing columns as 0.0
            df_c = df_c.reindex(columns=cols, fill_value=0.0).astype("float32")

            base = os.path.join(out_dir, f"cluster_{int(cid)}_training_data")
            # save matrix
            df_c.to_pickle(f"{base}.pkl")

            # save unit/columns manifest
            with open(f"{base}.units.json", "w") as f:
                json.dump({"unit":"g", "columns": cols, "meal_count": int(mask.sum())}, f)


    def save_cached_results(self, cache_file=None):
        """Save clustering results to disk."""
        if cache_file is None:
            cache_file = os.path.join(self.output_dir, "cached_clustering.pkl")
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        
        cache_data = {
            'meal_type_clusters': self.meal_type_clusters,
            'meal_type_features': self.meal_type_features,
            'meal_type_metadata': self.meal_type_metadata,
            'meal_type_stats': self.meal_type_stats,
            'significant_clusters': self.significant_clusters,
            'cluster_profiles': self.cluster_profiles
        }
        
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Cached clustering results saved to: {cache_file}")

    def load_cached_results(self, cache_file=None):
        """Load clustering results from disk."""
        if cache_file is None:
            cache_file = os.path.join(self.output_dir, "cached_clustering.pkl")
        if not os.path.exists(cache_file):
            print(f"Cache file not found: {cache_file}")
            return False
        
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Restore all cached data
            self.meal_type_clusters = cache_data['meal_type_clusters']
            self.meal_type_features = cache_data['meal_type_features']
            self.meal_type_metadata = cache_data['meal_type_metadata']
            self.meal_type_stats = cache_data['meal_type_stats']
            self.significant_clusters = cache_data['significant_clusters']
            self.cluster_profiles = cache_data.get('cluster_profiles', {})
            
            print(f"Loaded cached clustering results from: {cache_file}")
            return True
            
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False
    
    def clear_feature_cache(self, feature_type=None, normalize_method=None):
        """
        Clear cached features to force recomputation.
        
        Args:
            feature_type: Specific feature type to clear, or None for all
            normalize_method: Specific normalization method to clear, or None for all
        """
        cache_dir = self.output_dir
        
        if not os.path.exists(cache_dir):
            print("No cache directory found")
            return
        
        if feature_type is None and normalize_method is None:
            # Clear all feature caches
            cache_files = [f for f in os.listdir(cache_dir) if f.startswith('cached_features_')]
            for cache_file in cache_files:
                os.remove(os.path.join(cache_dir, cache_file))
                print(f"Removed cache file: {cache_file}")
        else:
            # Clear specific cache
            if feature_type and normalize_method:
                cache_path = self._get_feature_cache_path(feature_type, normalize_method)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    print(f"Removed cache file: {cache_path}")
                else:
                    print(f"Cache file not found: {cache_path}")
            else:
                print("Please specify both feature_type and normalize_method, or leave both as None to clear all")
    
    def get_cache_info(self):
        """Get information about cached features."""
        cache_dir = self.output_dir
        
        if not os.path.exists(cache_dir):
            print("No cache directory found")
            return
        
        cache_files = [f for f in os.listdir(cache_dir) if f.startswith('cached_features_')]
        
        if not cache_files:
            print("No cached feature files found")
            return
        
        print("Cached feature files:")
        for cache_file in cache_files:
            cache_path = os.path.join(cache_dir, cache_file)
            file_size = os.path.getsize(cache_path) / (1024 * 1024)  # MB
            file_age = time.time() - os.path.getmtime(cache_path)
            
            print(f"  {cache_file}:")
            print(f"    Size: {file_size:.1f} MB")
            print(f"    Age: {file_age/3600:.1f} hours")
            
            # Try to load cache info
            try:
                cached_data = pd.read_pickle(cache_path)
                if 'meal_counts' in cached_data:
                    print(f"    Meals: {cached_data['meal_counts']}")
                if 'timestamp' in cached_data:
                    print(f"    Created: {time.ctime(cached_data['timestamp'])}")
            except:
                print("    Could not read cache info")
