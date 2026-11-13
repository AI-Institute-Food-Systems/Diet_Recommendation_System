"""
Preprocessing (`make preprocess`, run from the repository root): NHANES 24-hour dietary recalls
to meal x food gram matrices per meal type.

main() runs these steps; a step whose output file already exists is loaded instead of rerun.
1. standardize_food_codes: read the NHANES individual-foods files (cached in data/raw/meals/),
   replace discontinued USDA food codes with current ones and split the recalls into breakfast,
   lunch and dinner meals.
2. build_systematic_prototype_mapping: map similar foods to nutrient-based prototype foods
   (data/mappings/food_to_prototype.pkl).
3. Per meal type, create_meal_df (meal x food matrices before and after the prototype mapping)
   and preprocessing(): LOF outlier removal, the bootstrap prevalence filter
   (<meal>_bci_filtered.pkl, the input to clustering) and the sparsity cut-off
   (<meal>_sparsity_filtered.pkl, with diagnostics in data/processed/real_meals/diagnostics/).
4. preprocessing_summary (a table in the log) and a macronutrient radar chart,
   data/processed/real_meals/diagnostics/nutrient_target_rdi.svg.

Outputs go to data/processed/real_meals/, data/mappings/ and logs/preprocessing/preprocessing.log;
the parsed recall files are cached in data/raw/meals/dataframes/<meal>_raw.pkl.
"""
import os
import re
import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from prettytable import PrettyTable
import pyreadstat

# pandas 2.x removed pandas.core.indexes.numeric (Int64Index, Float64Index, ...), which
# pickles written with pandas 1.x refer to. Install a stand-in module only when pandas
# lacks those classes; on pandas 1.x (the pinned version) the real module is used, and
# replacing it would break pickling of its index classes.
import importlib
import types
try:
    _real_numeric = importlib.import_module('pandas.core.indexes.numeric')
    _has_numeric_index_classes = all(
        hasattr(_real_numeric, name)
        for name in ('Int64Index', 'Float64Index', 'UInt64Index', 'NumericIndex')
    )
except Exception:
    _has_numeric_index_classes = False

if not _has_numeric_index_classes:
    numeric_module = types.ModuleType('pandas.core.indexes.numeric')
    numeric_module.NumericIndex = pd.Index
    numeric_module.Int64Index = pd.Index  # Use base Index as fallback
    numeric_module.Float64Index = pd.Index
    numeric_module.UInt64Index = pd.Index
    sys.modules['pandas.core.indexes.numeric'] = numeric_module
    import pandas.core.indexes as idx
    idx.numeric = numeric_module

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import Meal, download_file, fetch_data, setup_logger, to_pickle


def _normalize_index_for_pickle(df):
    """Rebuild the index and columns as plain pd.Index so that the pickle also loads under
    pandas 2.x, which has no Int64Index/Float64Index."""
    try:
        if hasattr(df, "index"):
            df.index = pd.Index(df.index.values)
        if hasattr(df, "columns"):
            df.columns = pd.Index(df.columns.values)
    except Exception:
        pass
    return df


def safe_to_pickle(data, filename):
    """Pickle a DataFrame after rebuilding its index and columns (see above)."""
    if hasattr(data, "index"):
        data = _normalize_index_for_pickle(data)
    return to_pickle(data, filename)
from evaluation.select_sparsity import choose_sparsity_threshold
from evaluation.plot_sparsity_dashboard import plot_sparsity_dashboard, plot_utility_vs_K_dashboard

import matplotlib.pyplot as plt

# Axes of the radar chart (plot_spider_chart). A set, so the axis order depends on string hashing.
# A tuple, not a set: the radar chart's axes follow this order, and a set's order changes with
# Python's hash seed from run to run.
MACRONUTRIENTS = (
    'Carbohydrate (g)',
    'Total Fat (g)',
    'Protein (g)',
    'Fiber (total dietary) (g)',
    'Monounsaturated fats (g)',
    'Polyunsaturated fats (g)',
    'Saturated fats (g)',
    'Stearic Acid (18:0) (g)',
    'Energy (kcal)',
)

# USDA nutrient column names -> names used here. Only the names change: no unit conversion is
# applied anywhere in this module, so e.g. 'Vitamin A (mg_RAE)' still holds mcg RAE values.
NUTRIENT_NAME_MAPPING = {
    'Fiber, total dietary (g)': 'Fiber (total dietary) (g)',
    'Fatty acids, total saturated (g)': 'Saturated fats (g)',
    'Fatty acids, total monounsaturated (g)': 'Monounsaturated fats (g)',
    'Fatty acids, total polyunsaturated (g)': 'Polyunsaturated fats (g)',
    '20:5 n-3 (g)': 'EPA (20:5 n-3) (g)',
    '22:6 n-3 (g)': 'DHA (22:6 n-3) (g)',
    '18:0 (g)': 'Stearic Acid (18:0) (g)',
    'Choline, total (mg)': 'Choline (mg)',
    'Vitamin A, RAE (mcg_RAE)': 'Vitamin A (mg_RAE)',
    'Vitamin D (D2 + D3) (mcg)': 'Vitamin D (IU)',
    'Vitamin K (phylloquinone) (mcg)': 'Vitamin K (mg)',
    'Folate, total (mcg)': 'Folate (mg_DFE)'
}


import pickle
from numpy.linalg import norm

# -------- Prototype merge knobs (systematic) --------
PROTOTYPE_ALPHA = 0.10         # scale K by subcat size (initial cap)
PROTOTYPE_K_MAX = 8            # hard cap per (sub)category
PROTOTYPE_MIN_SUBCAT = 6       # too small? roll to Main
PROTOTYPE_MASS_COVER = 0.90    # γ: anchors must cover >= 90% of weighted mass
PROTOTYPE_ERR_BUDGET = 0.07    # ε: weighted error (1 - cosine, see PROTOTYPE_ERR_MODE) must be <= 0.07

# ----- Weighting & allocation knobs -----
FORCE_PRESENCE_SEEDS = 1        # top-N presence foods per block, passed as seed_codes (unused by _auto_pick_K)

# Error/coverage controls
PROTOTYPE_ERR_MODE = 'cosine'   # 'cosine' or 'z_mare'
PROTOTYPE_SIM_FLOOR = 0.70      # cosine floor for coverage and for accepting an assignment

# Dynamic K allocation across subcategories
K_ALLOCATION_TOTAL = 1000       # soft total budget of prototypes across subcats
K_ALLOCATION_GAMMA = 1.0        # >1 emphasizes big/heterogeneous subcats, <1 flattens
K_ALLOCATION_BETA  = (0.5, 0.3, 0.2)  # (mass_share, presence_share, diversity)
K_MIN_PER_SUBCAT   = 1
K_MAX_PER_SUBCAT   = 32

# Input mappings (relative to the repository root) and the prototype mapping written here
FOOD_CATEGORY_MAP_PATH = 'data/mappings/food_ingredient_map.pkl'
FOOD_NUTRIENT_MAP_PATH = 'data/mappings/food_to_nutrient.pkl'
PROTOTYPE_MAPPING_PATH = 'data/mappings/food_to_prototype.pkl'

# Nutrients for the subcategory diversity term of allocate_K_per_subcat
NUTRIENT_SET = [
    'Energy (kcal)', 'Protein (g)', 'Total Fat (g)', 'Carbohydrate (g)',
    'Fiber (total dietary) (g)', 'Sugars, total (g)', 'Saturated fats (g)',
    'Sodium (mg)', 'Potassium (mg)'
]

def create_meals(data_subset, day_prefix):
    grouped = data_subset.groupby('SEQN')
    meals = []
    for _, group in grouped:
        id = group['SEQN'].iloc[0]
        food_codes = group[f'{day_prefix}IFDCD'].tolist()
        gram_portions = group[f'{day_prefix}IGRMS'].tolist()
        day_eaten = group[f'{day_prefix}DAY'].iloc[0]  # Assuming the same for all rows in the group.
        food_source = group[f'{day_prefix}FS'].tolist()
        meals.append(Meal(id, food_codes, gram_portions, day_eaten, food_source))
    return meals

def standardize_column_names(df, day_prefix):
    standardized_columns = {col: col.replace(day_prefix, '') for col in df.columns if col.startswith(day_prefix)}
    df.rename(columns=standardized_columns, inplace=True)
    return df

def get_lowest_food_codes(discontinued_id_map, code):
    code_info = discontinued_id_map.get(code)
    # if no entry for this code, it's a lowest code
    if code_info is None:
        return [{'code': code, 'revision': 0}]
    lowest_codes = []
    for new_code in code_info['code']: 
        lower_codes = get_lowest_food_codes(discontinued_id_map, new_code)
        if not lower_codes:  # if no lower codes, new code is lowest
            lowest_codes.append({'code': new_code, 'revision': code_info['revision']})
        else:
            lowest_codes.extend(lower_codes)  # otherwise add lower codes found

    return lowest_codes

def process_diet_data(data, day_prefix):
    breakfast_values = [1, 10]
    lunch_values = [2, 11]
    dinner_values = [3, 4, 14]

    breakfast_data = data[data[f'{day_prefix}_030Z'].isin(breakfast_values)]
    lunch_data = data[data[f'{day_prefix}_030Z'].isin(lunch_values)]
    dinner_data = data[data[f'{day_prefix}_030Z'].isin(dinner_values)]

    breakfast_meals = create_meals(breakfast_data, day_prefix)
    lunch_meals = create_meals(lunch_data, day_prefix)
    dinner_meals = create_meals(dinner_data, day_prefix)

    return breakfast_meals, lunch_meals, dinner_meals

def generate_discontinued_id_map(discontinued_urls, cache_dir, preprocess_log):
    discontinued_id_map = {}
    for file_url in discontinued_urls:
        try:
            filename = os.path.basename(file_url)
            local_filepath = os.path.join(cache_dir, filename)
            if not os.path.exists(local_filepath):
                preprocess_log.info(f"Downloading {filename} from {file_url}...")
                download_file(file_url, local_filepath, preprocess_log)
            else:
                preprocess_log.info(f"Using cached file: {filename}")

            if local_filepath.endswith(".xlsx"):
                discontinued_df = pd.read_excel(local_filepath, engine='openpyxl', skiprows=1)
            else:
                raise ValueError(f"Unsupported file format for {local_filepath}")

            # Get the first and fourth columns as the old and new food codes
            old_code_column = discontinued_df.columns[0]
            new_code_column = discontinued_df.columns[3]
            discontinued_id_column = discontinued_df.columns[2] 

            for _, row in discontinued_df.iterrows():
                discontinued_id = row[discontinued_id_column]
                old_code = row[old_code_column]
                new_codes = row[new_code_column]
                # Handle the case where there are multiple new codes for a single old code.
                if discontinued_id == 2 and isinstance(new_codes, str):
                    new_codes = re.split(r'[;\n]', new_codes)
                    new_codes = [int(code.strip()) for code in new_codes if code.strip()]
                elif discontinued_id == 5 or discontinued_id == 1:
                    # For dropped (1) or revised (5) items, keep original code but still add to map
                    new_codes = [old_code]  # Keep the same code
                else: # discontinued codes 3 or 4
                    new_codes = [int(new_codes)]

                discontinued_id_map[old_code] = {
                    'code': new_codes,  
                    'revision': discontinued_id, 
                }
        except Exception as e:
            preprocess_log.error(f"Error processing file {filename}: {e}")
            continue
        
    to_pickle(discontinued_id_map, 'data/mappings/discontinued_food_codes.pkl')
    return discontinued_id_map

def process_row(row, day_prefix, discontinued_id_map):
    food_code = row[f'{day_prefix}IFDCD']
    lowest_codes = get_lowest_food_codes(discontinued_id_map, food_code)
    
    if not lowest_codes:
        return [row]
    
    updated_rows = []
    for code_info in lowest_codes:
        new_code = code_info['code']
        discontinued_id = code_info['revision']

        new_row = row.copy()
        if discontinued_id == 1 or discontinued_id == 5:
            # for dropped (1) or revised (5) items, we keep the original code.
            updated_rows.append(new_row)
        elif discontinued_id == 2:
            # for expanded (2) items, we create new rows with distributed gram weights
            portion = row[f'{day_prefix}IGRMS'] / len(lowest_codes)  # Divide equally among new codes
            new_row[f'{day_prefix}IFDCD'] = new_code
            new_row[f'{day_prefix}IGRMS'] = portion
            updated_rows.append(new_row)
        else:
            # for consolidated (3) or renumbered (4) items, we replace with the new code
            new_row[f'{day_prefix}IFDCD'] = new_code
            updated_rows.append(new_row)
    return updated_rows

def update_food_codes(day_prefix, data, discontinued_id_map, preprocess_log):
    original_unique_food_codes = data[f'{day_prefix}IFDCD'].unique()
    original_food_code_count = data[f'{day_prefix}IFDCD'].nunique()
    
    updated_data_list = data.apply(lambda row: process_row(row, day_prefix, discontinued_id_map), axis=1)
    flat_list = [item for sublist in updated_data_list for item in sublist]
    updated_data = pd.DataFrame(flat_list)

    updated_food_code_count = updated_data[f'{day_prefix}IFDCD'].nunique()
    overall_difference = updated_food_code_count - original_food_code_count
    preprocess_log.info(f"Original unique food codes:       {original_food_code_count}")
    preprocess_log.info(f"Updated unique food codes:        {updated_food_code_count}")
    preprocess_log.info(f"Overall difference in food codes: {overall_difference}")
    
    return updated_data, original_unique_food_codes

def merge_weight_columns(df):
    if 'WTDRD1' in df.columns:
        df['WTD1'] = df['WTDRD1']
        df.drop(['WTDRD1'], axis=1, inplace=True)
    if 'WTDRD1PP' in df.columns:
        df['WTD1'] = df['WTDRD1PP']
        df.drop(['WTDRD1PP'], axis=1, inplace=True)
    if 'WTDR2D' in df.columns:
        df['WTD2'] = df['WTDR2D']
        df.drop(['WTDR2D'], axis=1, inplace=True)
    if 'WTDR2DPP' in df.columns:
        df['WTD2'] = df['WTDR2DPP']
        df.drop(['WTDR2DPP'], axis=1, inplace=True)
    return df

def load_and_process_meal_data(meal_survey_urls, discontinued_id_map, cache_dir, preprocess_log):
    all_updated_data, original_unique_food_codes = [], []
    breakfast_meals, lunch_meals, dinner_meals = [], [], []
    breakfast_raw_meals, lunch_raw_meals, dinner_raw_meals = [], [], []
    for url in meal_survey_urls:
        
        try:
            preprocess_log.info(f"Processing data from URL: {url}")
            file_path = fetch_data(url, cache_dir, preprocess_log)
            # Read the XPT file into a pandas DataFrame
            meal_data, _ = pyreadstat.read_xport(file_path)
            preprocess_log.info(f"Unique Respondents: {meal_data['SEQN'].nunique()}")
            preprocess_log.info(f"Number of Records: {len(meal_data)}")
            day_prefix = 'DR1' if 'DR1' in url else 'DR2'

            breakfast_raw, lunch_raw, dinner_raw = process_diet_data(meal_data, day_prefix)
            breakfast_raw_meals.extend(breakfast_raw)
            lunch_raw_meals.extend(lunch_raw)
            dinner_raw_meals.extend(dinner_raw)
            
            preprocess_log.info(f"Raw meals: {len(breakfast_raw) + len(lunch_raw) + len(dinner_raw)} - {len(breakfast_raw)} breakfast, {len(lunch_raw)} lunch, and {len(dinner_raw)} dinner from {url}")
            updated_meal_data, subset_unique_food_codes = update_food_codes(day_prefix, meal_data, discontinued_id_map, preprocess_log)
            original_unique_food_codes.extend(subset_unique_food_codes)
            updated_meal_data = updated_meal_data[updated_meal_data[f'{day_prefix}DRSTZ'] == 1]  # diet recall status code is 1
            updated_meal_data = updated_meal_data[updated_meal_data['DRABF'] == 2]  # breast-fed infant is 'No'
            
            breakfast_temp, lunch_temp, dinner_temp = process_diet_data(updated_meal_data, day_prefix)
            preprocess_log.info(f"Processed meals: {len(breakfast_temp) + len(lunch_temp) + len(dinner_temp)} - {len(breakfast_temp)} breakfast, {len(lunch_temp)} lunch, and {len(dinner_temp)} dinner from {url}")
            # Merge the weight columns before standardizing column names
            updated_meal_data = merge_weight_columns(updated_meal_data)
            updated_meal_data = standardize_column_names(updated_meal_data, day_prefix)
            all_updated_data.append(updated_meal_data)
            breakfast_meals.extend(breakfast_temp)
            lunch_meals.extend(lunch_temp)        
            dinner_meals.extend(dinner_temp)
            
        except Exception as e:
            preprocess_log.error(f"Error processing meal data from {url}: {e}")
            continue
        
    raw_df_dir = 'data/raw/meals/dataframes/'
    os.makedirs(raw_df_dir, exist_ok=True)
    to_pickle(breakfast_raw_meals, raw_df_dir+'breakfast_raw.pkl')
    to_pickle(lunch_raw_meals, raw_df_dir+'lunch_raw.pkl')
    to_pickle(dinner_raw_meals, raw_df_dir+'dinner_raw.pkl')
    
    preprocess_log.info(f"Raw unique food codes: {len(set(original_unique_food_codes))}")
    combined_meal_data = pd.concat(all_updated_data, ignore_index=True)
    to_pickle(combined_meal_data, 'data/processed/real_meals/dataframes/combined_food_codes_standardized.pkl')
    to_pickle(breakfast_meals, 'data/processed/real_meals/breakfast_meals.pkl')
    to_pickle(lunch_meals, 'data/processed/real_meals/lunch_meals.pkl')
    to_pickle(dinner_meals, 'data/processed/real_meals/dinner_meals.pkl')
    
    return combined_meal_data, breakfast_meals, lunch_meals, dinner_meals

def remove_outliers(df, meal_type, preprocess_log, contamination=0.003):
    """Drop meals whose Local Outlier Factor score (20 neighbours) is above the 99.7th
    percentile and save <meal>_outliers_removed.pkl. The cut-off is fixed at 99.7 %;
    `contamination` only sets the labels of fit_predict, which are not used."""
    try:
        preprocess_log.info("Removing outliers...")
        X = df.values
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=False)
        # Where -1 indicates outliers and 1 indicates inliers
        lof.fit_predict(X)
        # Calculate LOF scores (negative_outlier_factor_ is opposite, more negative means more outlier)
        lof_scores = -lof.negative_outlier_factor_  # We negate to make positive, higher means more outlier
        cutoff_score = np.percentile(lof_scores, 99.7)
        # Identifying indexes to keep (inliers and low-scoring outliers)
        inlier_indexes = (lof_scores <= cutoff_score)
        num_outliers = np.sum(~inlier_indexes)
        preprocess_log.info(f"Number of outliers removed: {num_outliers}")
        filtered_df = df[inlier_indexes]
        safe_to_pickle(filtered_df, f'data/processed/real_meals/dataframes/{meal_type}_outliers_removed.pkl')
    except Exception as e:
        preprocess_log.error("Error in removing outliers: " + str(e))
    return filtered_df

def bootstrap_confidence_interval_filter(df, meal_type, preprocess_log, n_iterations=1000, ci_threshold=0.95):
    """Keep the foods whose presence rate is above the lower bound of its bootstrap CI
    (n_iterations resamples of the meals), then drop meals left without foods; saves
    <meal>_bci_filtered.pkl. The resampling uses NumPy's global RNG without a seed, but in
    practice only foods that never occur fail the test, so the output does not change."""
    try:
        preprocess_log.info("Applying bootstrap confidence interval filter...")
        binary_df = df.gt(0).astype(int)
        bootstrapped_means = np.zeros((n_iterations, binary_df.shape[1]))
        
        for i in range(n_iterations):
            sample = binary_df.sample(frac=1.0, replace=True)
            bootstrapped_means[i, :] = sample.mean()
        lower_bounds = np.percentile(bootstrapped_means, (1 - ci_threshold) / 2 * 100, axis=0)

        # Determine which food codes are above the lower bound of the CI
        mask = np.mean(binary_df, axis=0) > lower_bounds
        filtered_df = df.loc[:, mask]
        
        foods_retained = np.sum(mask)
        foods_removed = df.shape[1] - foods_retained
        preprocess_log.info(f"Retained {foods_retained} foods, removed {foods_removed} foods")
        
        num_rows_before = filtered_df.shape[0]
        filtered_df = filtered_df[filtered_df.any(axis=1)]
        num_rows_after = filtered_df.shape[0]
        num_rows_removed = num_rows_before - num_rows_after
        preprocess_log.info(f"Removed {num_rows_removed} meals with all zero values post-filtration")

        safe_to_pickle(filtered_df, f'data/processed/real_meals/dataframes/{meal_type}_bci_filtered.pkl')
    except Exception as e:
        preprocess_log.error("Error in bootstrap confidence interval filtering: " + str(e))

def preprocessing(meal_type, preprocess_log):
    """Outlier removal, bootstrap prevalence filter and sparsity cut-off for one meal type,
    starting from <meal>_prototype_mapped.pkl; each step is skipped if its output exists."""
    try:
        preprocess_log.info(f"Preprocessing for meal type: {meal_type}")
        outlier_file = f'data/processed/real_meals/dataframes/{meal_type}_outliers_removed.pkl'
        bci_file = f'data/processed/real_meals/dataframes/{meal_type}_bci_filtered.pkl'
        sparsity_file = f'data/processed/real_meals/dataframes/{meal_type}_sparsity_filtered.pkl'

        if not os.path.exists(outlier_file):
            # Use prototype_mapped data as input for preprocessing
            meal_df = pd.read_pickle(f'data/processed/real_meals/dataframes/{meal_type}_prototype_mapped.pkl')
            meal_df = _normalize_index_for_pickle(meal_df)
            outlier_df  = remove_outliers(meal_df, meal_type, preprocess_log)
        else:
            preprocess_log.info(f'{outlier_file} exists, using cached results.')
            outlier_df = pd.read_pickle(outlier_file)
            outlier_df = _normalize_index_for_pickle(outlier_df)
        
        if not os.path.exists(bci_file):
            bootstrap_confidence_interval_filter(outlier_df, meal_type, preprocess_log)
            bci_df = pd.read_pickle(bci_file)
        else:
            preprocess_log.info(f'{bci_file} exists, using cached results.')
            bci_df = pd.read_pickle(bci_file)
            
        if not os.path.exists(sparsity_file):
            preprocess_log.info("Selecting sparsity operating point (no model)...")
            best_pct, metrics_df, preprocessed_data = choose_sparsity_threshold(
                bci_df,
                percent_grid=tuple(range(0, 100, 5)),
                min_meal_retain=0.98,           # keep >= 98% of the meals
                min_median_nonzeros=3,          # median meal keeps >= 3 foods
                min_utility_retain=0.70,        # keep >= 70% of the Gini utility
                strong_mode=False,               # knee point, clamped to the passing range
                p_min_abs=0.002,                # food in >= 0.2% of meals
                min_occurrences_frac=0.001,     # food in >= 0.1% of meals (count)
                foods_ratio=0.80,               # keep at most 80% of the foods
            )
            diag_dir = 'data/processed/real_meals/diagnostics'
            # Generate both dashboards
            plot_sparsity_dashboard(metrics_df, best_pct, meal_type, diag_dir)
            plot_utility_vs_K_dashboard(metrics_df, best_pct, meal_type, diag_dir, bci_df)

            os.makedirs(diag_dir, exist_ok=True)
            metrics_df.to_csv(f'{diag_dir}/{meal_type}_sparsity_metrics.csv', index=False)

            preprocess_log.info(f"Chosen sparsity percentile: {best_pct}")
            preprocess_log.info(
                f"Sparsity filtered dataset has {preprocessed_data.shape[0]} meals and {preprocessed_data.shape[1]} foods"
            )
            safe_to_pickle(preprocessed_data, f'data/processed/real_meals/dataframes/{meal_type}_sparsity_filtered.pkl')

        else:
            preprocess_log.info(f'{sparsity_file} exists, using cached results.')

    except Exception as e:
        preprocess_log.error("Error in preprocessing: " + str(e))
        

def create_meal_df(meal_type, preprocess_log):
    """Meal x food gram matrices for one meal type from <meal>_meals.pkl, keeping only foods
    with nutrient and category data: <meal>_food_codes_standardized.pkl (original codes) and
    <meal>_prototype_mapped.pkl (foods rolled up with the prototype mapping, if present)."""
    try:
        preprocess_log.info(f"Creating meal DataFrame for: {meal_type}")
        meal_list = pd.read_pickle(f'data/processed/real_meals/{meal_type}_meals.pkl')

        # First, create DataFrame without prototype mapping
        preprocess_log.info("Creating initial DataFrame without prototype mapping...")
        meal_dicts_raw = []
        for meal in meal_list:
            meal_dicts_raw.append(dict(zip(meal.food_codes, meal.gram_portions)))

        meal_df_raw = pd.DataFrame(meal_dicts_raw).fillna(0)
        # Preliminary filter: keep only foods that have nutrient & category mappings
        try:
            import pickle as _pkl
            cat_map = _pkl.load(open(FOOD_CATEGORY_MAP_PATH, 'rb')) if os.path.exists(FOOD_CATEGORY_MAP_PATH) else None
            nut_map = pd.read_pickle(FOOD_NUTRIENT_MAP_PATH) if os.path.exists(FOOD_NUTRIENT_MAP_PATH) else None
            allowed_nut = set()
            if nut_map is not None:
                if 'Food code' in nut_map.columns:
                    allowed_nut = set(pd.to_numeric(nut_map['Food code'], errors='coerce').dropna().astype(int).tolist())
                else:
                    # fallback: first numeric-like column
                    for c in nut_map.columns:
                        if pd.api.types.is_numeric_dtype(nut_map[c]):
                            allowed_nut = set(pd.to_numeric(nut_map[c], errors='coerce').dropna().astype(int).tolist())
                            break
            allowed_cat = set()
            if cat_map is not None and 'Food code' in cat_map.columns:
                allowed_cat = set(pd.to_numeric(cat_map['Food code'], errors='coerce').dropna().astype(int).tolist())
            # intersection if both available, else whichever exists
            allowed = allowed_nut & allowed_cat if allowed_nut and allowed_cat else (allowed_nut or allowed_cat)
            if allowed:
                cols_before = meal_df_raw.shape[1]
                # normalize column labels to int where possible for comparison
                col_ints = []
                for c in meal_df_raw.columns:
                    try:
                        col_ints.append(int(str(c).split('.')[0]))
                    except Exception:
                        col_ints.append(None)
                keep_mask = [ci in allowed for ci in col_ints]
                meal_df_raw = meal_df_raw.loc[:, keep_mask]
                removed = cols_before - meal_df_raw.shape[1]
                preprocess_log.info(f"Preliminary filter (have nutrient/category): kept={meal_df_raw.shape[1]}, removed={removed}")
            else:
                preprocess_log.warning("Preliminary filter skipped: could not derive allowed nutrient/category codes")
        except Exception as e:
            preprocess_log.warning(f"Preliminary filter error: {e}")

        to_pickle(meal_df_raw, f'data/processed/real_meals/dataframes/{meal_type}_food_codes_standardized.pkl')
        
        preprocess_log.info(f"Initial DataFrame: {meal_df_raw.shape[0]} meals, {meal_df_raw.shape[1]} unique foods")
        
        # Apply prototype mapping if available
        mapping = None
        if os.path.exists(PROTOTYPE_MAPPING_PATH):
            preprocess_log.info("Loading prototype mapping for roll-up...")
            with open(PROTOTYPE_MAPPING_PATH, 'rb') as f:
                mapping = pickle.load(f)

            preprocess_log.info("Applying prototype mapping...")
            # Build allowed code set (intersection of nutrient and category maps)
            try:
                import pickle as _pkl
                _cat_map = _pkl.load(open(FOOD_CATEGORY_MAP_PATH, 'rb')) if os.path.exists(FOOD_CATEGORY_MAP_PATH) else None
                _nut_map = pd.read_pickle(FOOD_NUTRIENT_MAP_PATH) if os.path.exists(FOOD_NUTRIENT_MAP_PATH) else None
                _allowed_nut = set()
                if _nut_map is not None:
                    if 'Food code' in _nut_map.columns:
                        _allowed_nut = set(pd.to_numeric(_nut_map['Food code'], errors='coerce').dropna().astype(int).tolist())
                    else:
                        for _c in _nut_map.columns:
                            if pd.api.types.is_numeric_dtype(_nut_map[_c]):
                                _allowed_nut = set(pd.to_numeric(_nut_map[_c], errors='coerce').dropna().astype(int).tolist())
                                break
                _allowed_cat = set()
                if _cat_map is not None and 'Food code' in _cat_map.columns:
                    _allowed_cat = set(pd.to_numeric(_cat_map['Food code'], errors='coerce').dropna().astype(int).tolist())
                _allowed_codes = _allowed_nut & _allowed_cat if _allowed_nut and _allowed_cat else (_allowed_nut or _allowed_cat)
                if not _allowed_codes:
                    preprocess_log.warning("No allowed code set could be derived; prototype enforcement disabled")
            except Exception as _e:
                preprocess_log.warning(f"Error building allowed code set: {_e}")
                _allowed_codes = None

            meal_dicts_proto = []
            for meal in meal_list:
                meal_dicts_proto.append(_apply_prototype_mapping_to_meal(meal, mapping, _allowed_codes))

            meal_df_proto = pd.DataFrame(meal_dicts_proto).fillna(0)
            # Enforce allowed codes at dataframe level as an additional guard
            if _allowed_codes:
                try:
                    _cols_before = meal_df_proto.shape[1]
                    _col_ints = []
                    for _c in meal_df_proto.columns:
                        try:
                            _col_ints.append(int(str(_c).split('.')[0]))
                        except Exception:
                            _col_ints.append(None)
                    _keep = [(_ci in _allowed_codes) for _ci in _col_ints]
                    meal_df_proto = meal_df_proto.loc[:, _keep]
                    _removed = _cols_before - meal_df_proto.shape[1]
                    if _removed > 0:
                        preprocess_log.info(f"Prototype enforcement filter: removed {_removed} disallowed prototype columns")
                except Exception as _e:
                    preprocess_log.warning(f"Prototype enforcement filter error: {_e}")
            to_pickle(meal_df_proto, f'data/processed/real_meals/dataframes/{meal_type}_prototype_mapped.pkl')
            
            # Log the reduction effect and the most common prototype foods
            original_foods = meal_df_raw.shape[1]
            prototype_foods = meal_df_proto.shape[1]
            reduction_pct = (original_foods - prototype_foods) / original_foods * 100 if original_foods > 0 else 0

            preprocess_log.info("Prototype mapping effect:")
            preprocess_log.info(f"  - Original foods: {original_foods}")
            preprocess_log.info(f"  - Prototype foods: {prototype_foods}")
            preprocess_log.info(f"  - Reduction: {original_foods - prototype_foods} foods ({reduction_pct:.1f}%)")
            preprocess_log.info(f"  - Meals: {meal_df_proto.shape[0]} (unchanged)")

            # Log the prototype columns (first N) and their meal counts
            try:
                col_counts = (meal_df_proto > 0).sum(axis=0).sort_values(ascending=False)
                top_n = 50
                preprocess_log.info(f"  - Top {top_n} prototype foods by nonzero meals:")
                for food_code, cnt in col_counts.head(top_n).items():
                    preprocess_log.info(f"      {food_code}: {int(cnt)} meals")
            except Exception as e:
                preprocess_log.warning(f"Could not log the prototype table: {e}")
            
        else:
            preprocess_log.info("No prototype mapping found, using original food codes")
            # If no mapping, copy the standardized file to prototype_mapped for consistency
            to_pickle(meal_df_raw, f'data/processed/real_meals/dataframes/{meal_type}_prototype_mapped.pkl')
            
    except Exception as e:
        preprocess_log.error("Error in creating meal DataFrame: " + str(e))

def preprocessing_summary(meal_types, processing_filetypes, preprocess_log):
    preprocess_log.info("="*66 + " Preprocessing Summary " + "="*67)
    summary_table = PrettyTable()
    summary_table.field_names = ["Processing Step"] + [f"{meal_type.capitalize()} Meals" for meal_type in meal_types] + ["Unique Foods"] + [f"{meal_type.capitalize()} Foods" for meal_type in meal_types] 
    for processing_step in processing_filetypes:
        process_meals, process_foods = [], []
        total_foods = []
        for meal_type in meal_types:
            if processing_step == 'raw':
                try:
                    data = pd.read_pickle(f'data/raw/meals/dataframes/{meal_type}_raw.pkl')
                    meal_type_foods = []
                    for meal in data:
                        meal_type_foods.extend(meal.food_codes)
                    unique_foods = np.unique(meal_type_foods).tolist()
                    process_meals.append(len(data))
                    process_foods.append(len(unique_foods))
                    total_foods.extend(unique_foods)
                except FileNotFoundError:
                    preprocess_log.warning(f"Raw data file for {meal_type} not found: data/raw/meals/dataframes/{meal_type}_raw.pkl")
                    process_meals.append("N/A")
                    process_foods.append("N/A")
            else:
                try:
                    data = pd.read_pickle(f'data/processed/real_meals/dataframes/{meal_type}_{processing_step}.pkl')
                    unique_foods = data.columns
                    process_meals.append(data.shape[0])
                    process_foods.append(data.shape[1])
                    total_foods.extend(np.unique(unique_foods).tolist())
                except FileNotFoundError:
                    preprocess_log.warning(
                        f"Processed data file for {meal_type} not found: "
                        f"data/processed/real_meals/dataframes/{meal_type}_{processing_step}.pkl"
                    )
                    process_meals.append("N/A")
                    process_foods.append("N/A")
        if total_foods:
            total_foods = len(np.unique(total_foods))
        else:
            total_foods = "N/A"
        summary_table.add_row([processing_step] + process_meals + [total_foods] + process_foods) 

    table_string = summary_table.get_string()
    for line in table_string.splitlines():
        preprocess_log.info(line)


def _cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T

def _prep_nutrient_matrix(food_nutrient_mapping: pd.DataFrame,
                          category_map: pd.DataFrame) -> pd.DataFrame:
    """Z-scored nutrient matrix indexed by food code, with the WWEIA 'Main' and 'Subcat'
    columns; also sets the module-level ACTIVE_NUTRIENTS used by the prototype search."""
    # 1) normalize nutrient cols
    df = food_nutrient_mapping.copy()
    df.columns = df.columns.str.replace('\n', ' ')
    df.rename(columns=NUTRIENT_NAME_MAPPING, inplace=True)

    # --- detect + validate nutrients ---
    detected = _detect_nutrient_columns(food_nutrient_mapping)
    # Build W early so validation can work on the joined frame
    cols = ['Food code'] + [c for c in detected if c in df.columns]
    W = df[cols].drop_duplicates(subset=['Food code']).set_index('Food code')

    # 2) De-dup category map per food code
    meta_raw = category_map[['Food code',
                             'WWEIA Main Category description',
                             'WWEIA Subcategory description']].copy()

    def _mode_or_first(s: pd.Series):
        m = s.mode(dropna=True)
        if len(m) > 0:
            return m.iloc[0]
        s = s.dropna()
        return s.iloc[0] if len(s) else np.nan

    meta = (meta_raw
            .groupby('Food code', as_index=True)
            .agg({'WWEIA Main Category description': _mode_or_first,
                  'WWEIA Subcategory description': _mode_or_first})
            .rename(columns={'WWEIA Main Category description': 'Main',
                             'WWEIA Subcategory description': 'Subcat'}))

    W = W.join(meta, how='left')

    if not W.index.is_unique:
        W = W[~W.index.duplicated(keep='first')]

    # 3) Validate features against W (after join)
    features = _validate_nutrient_features(W, detected)

    # 4) impute nutrients: subcat median -> main median -> 0
    for c in features:
        med_sub = W.groupby('Subcat', dropna=False)[c].transform('median')
        W[c] = W[c].fillna(med_sub)
        med_main = W.groupby('Main', dropna=False)[c].transform('median')
        W[c] = W[c].fillna(med_main).fillna(0.0)

    # 5) mild de-skew on a couple of minerals if present
    for c in ['Sodium (mg)', 'Potassium (mg)']:
        if c in features:
            W[c] = np.log1p(W[c])

    # 6) z-score on the final feature set only
    Z = W[features].astype(float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    Z['Main'] = W['Main']
    Z['Subcat'] = W['Subcat']

    Z.index = Z.index.astype(int)
    assert Z.index.is_unique, "Z has non-unique Food code index after prep; check category map."

    # --- publish the features so every cosine/MARE uses the same vector ---
    global ACTIVE_NUTRIENTS
    ACTIVE_NUTRIENTS = features

    return Z


def _food_weights_from_rows(combined_rows: pd.DataFrame,
                            alpha: float = 0.7) -> pd.Series:
    # grams
    wtcol = 'WTD1' if 'WTD1' in combined_rows.columns else ('WTD2' if 'WTD2' in combined_rows.columns else None)
    grams = (combined_rows['IGRMS'] * combined_rows[wtcol].fillna(1.0) if wtcol else combined_rows['IGRMS'])
    grams_by_food = grams.groupby(combined_rows['IFDCD'].astype(int)).sum()

    # presence (unique SEQN–DAY occurrences per food)
    if all(c in combined_rows.columns for c in ['SEQN','DAY','IFDCD']):
        pres = (combined_rows[['SEQN','DAY','IFDCD']]
                .drop_duplicates()
                .groupby('IFDCD').size().astype(float))
    else:
        pres = combined_rows.groupby('IFDCD').size().astype(float)
    pres.index = pres.index.astype(int)

    # normalize
    g = grams_by_food / (grams_by_food.sum() + 1e-12)
    p = pres / (pres.sum() + 1e-12)

    # blend and rescale back to "gram-like" magnitude (optional)
    w = alpha * g.add((1 - alpha) * p, fill_value=0.0)
    w = w * float(grams_by_food.sum())  # keeps coverage metric comparable
    return w


def _subcat_diversity(Z: pd.DataFrame) -> pd.Series:
    """
    Diversity per Subcat in z-space: mean(1 - cosine) within block, normalized to [0,1].
    """
    vals = {}
    for sub, D in Z.groupby('Subcat'):
        X = D[NUTRIENT_SET].values
        if len(D) <= 1:
            vals[sub] = 0.0
        else:
            S = _cosine_sim(X, X)
            np.fill_diagonal(S, np.nan)
            vals[sub] = float(1.0 - np.nanmean(S))
    s = pd.Series(vals)
    if s.max() > s.min():
        s = (s - s.min()) / (s.max() - s.min())
    else:
        s = s * 0.0
    return s


def allocate_K_per_subcat(cat_map: pd.DataFrame,
                          rows: pd.DataFrame,
                          Z: pd.DataFrame,
                          K_total: int = K_ALLOCATION_TOTAL,
                          gamma: float = K_ALLOCATION_GAMMA,
                          beta: tuple[float,float,float] = K_ALLOCATION_BETA,
                          kmin: int = K_MIN_PER_SUBCAT,
                          kmax: int = K_MAX_PER_SUBCAT) -> dict[str,int]:
    """
    Allocate a K per Subcat using a soft budget across subcats by importance:
       I_s = β1 * mass_share_s + β2 * presence_share_s + β3 * diversity_s
    Returns dict: {Subcat -> K_cap}
    """
    # ---------- 1) Build a UNIQUE FoodCode -> Subcat map ----------
    def _mode_or_first(s: pd.Series):
        m = s.mode(dropna=True)
        if len(m) > 0:
            return m.iloc[0]
        s = s.dropna()
        return s.iloc[0] if len(s) else np.nan

    submap = (cat_map[['Food code', 'WWEIA Subcategory description']]
              .dropna(subset=['Food code'])
              .groupby('Food code', as_index=True)['WWEIA Subcategory description']
              .agg(_mode_or_first))
    # ensure int index so it aligns with meal/code series
    try:
        submap.index = submap.index.astype(int)
    except Exception:
        submap.index = submap.index.astype(float).astype(int)

    # Universe of subcats present in Z
    subcats_universe = pd.Index(Z['Subcat'].dropna().unique())

    # ---------- 2) Mass share per subcat ----------
    wtcol = 'WTD1' if 'WTD1' in rows.columns else ('WTD2' if 'WTD2' in rows.columns else None)
    grams = (rows['IGRMS'] * rows[wtcol].fillna(1.0) if wtcol else rows['IGRMS'])
    grams_by_food = grams.groupby(rows['IFDCD'].astype(int)).sum()

    df_mass = (pd.DataFrame({'mass': grams_by_food})
               .join(submap.rename('Subcat'), how='left'))
    sub_mass = df_mass.groupby('Subcat')['mass'].sum().reindex(subcats_universe, fill_value=0.0)
    sub_mass = sub_mass / (sub_mass.sum() + 1e-12)

    # ---------- 3) Presence share per subcat (unique SEQN-DAY-food) ----------
    if all(c in rows.columns for c in ['SEQN','DAY','IFDCD']):
        pres = (rows[['SEQN','DAY','IFDCD']]
                .drop_duplicates()
                .groupby('IFDCD')
                .size()
                .astype(float))
    else:
        pres = rows.groupby('IFDCD').size().astype(float)
    pres.index = pres.index.astype(int)

    df_pres = (pd.DataFrame({'presence': pres})
               .join(submap.rename('Subcat'), how='left'))
    sub_pres = df_pres.groupby('Subcat')['presence'].sum().reindex(subcats_universe, fill_value=0.0)
    sub_pres = sub_pres / (sub_pres.sum() + 1e-12)

    # ---------- 4) Diversity per subcat (from Z) ----------
    sub_div = _subcat_diversity(Z).reindex(subcats_universe, fill_value=0.0)

    # ---------- 5) Importance & soft allocation ----------
    b1, b2, b3 = beta
    I = (b1 * sub_mass).add(b2 * sub_pres, fill_value=0.0).add(b3 * sub_div, fill_value=0.0)
    I = I.clip(lower=0.0)
    if I.sum() == 0:
        I = pd.Series(1.0, index=subcats_universe)

    w = I.pow(gamma)
    w = w / (w.sum() + 1e-12)

    # raw allocation
    K_soft = (w * float(K_total)).round().astype(int)

    # ---------- 6) Clip by availability & global caps ----------
    # available items per subcat in Z
    n_per_subcat = Z.groupby('Subcat').size().reindex(subcats_universe, fill_value=0).astype(int)
    K_soft = (K_soft
              .clip(lower=kmin)
              .clip(upper=kmax)
              .clip(upper=n_per_subcat))  # can't exceed available items

    return K_soft[K_soft > 0].to_dict()


def _facility_location(Z_block: pd.DataFrame, K: int, weights: np.ndarray | None) -> list:
    X = Z_block[ACTIVE_NUTRIENTS].values
    sims = _cosine_sim(X, X)
    np.fill_diagonal(sims, 1.0)
    n = sims.shape[0]
    w = np.asarray(weights) if weights is not None else np.ones(n, dtype=float)
    best = np.zeros(n, dtype=float)
    chosen = []
    for _ in range(min(K, n)):
        mg = (np.maximum(0.0, sims - best[:, None]) * w[:, None]).sum(axis=0)
        j = int(np.argmax(mg))
        chosen.append(Z_block.index[j])
        best = np.maximum(best, sims[:, j])
    return chosen


def _assign_hard_codes(Z_block: pd.DataFrame, prot_codes: list) -> tuple[np.ndarray, np.ndarray]:
    if len(prot_codes) == 0:
        idx = Z_block.index.to_numpy()
        return idx, np.zeros(len(idx), dtype=float)

    prot_codes = [p for p in prot_codes if p in Z_block.index]
    if len(prot_codes) == 0:
        idx = Z_block.index.to_numpy()
        return idx, np.zeros(len(idx), dtype=float)

    X = Z_block[ACTIVE_NUTRIENTS].values
    P = Z_block.loc[prot_codes, ACTIVE_NUTRIENTS].values
    S = _cosine_sim(X, P)
    j = np.argmax(S, axis=1)
    m = S[np.arange(S.shape[0]), j]
    prot_arr = np.asarray(prot_codes)
    assigned_codes = prot_arr[j]
    return assigned_codes, m


def _weighted_mare_from_codes(Z_block: pd.DataFrame, assigned_codes, weights):
    X = Z_block[ACTIVE_NUTRIENTS].values
    P = Z_block.loc[assigned_codes, ACTIVE_NUTRIENTS].values
    num = np.abs(X - P)
    den = np.abs(X) + 1e-6
    per_food = (num / den).mean(axis=1)
    w = weights if weights is not None else np.ones_like(per_food)
    return float(np.average(per_food, weights=w) if np.sum(w) > 0 else np.mean(per_food))


def _weighted_cosine_err(sim: np.ndarray, weights: np.ndarray) -> float:
    err = 1.0 - np.asarray(sim)
    w = np.asarray(weights)
    return float(np.average(err, weights=w) if w.sum() > 0 else np.mean(err))


def _compute_err(Z_block: pd.DataFrame,
                 assigned_codes: np.ndarray,
                 sim: np.ndarray,
                 weights: np.ndarray,
                 mode: str = PROTOTYPE_ERR_MODE) -> float:
    if mode == 'cosine':
        return _weighted_cosine_err(sim, weights)
    # fallback to original z-space MARE
    return _weighted_mare_from_codes(Z_block, assigned_codes, weights)


def _auto_pick_K(Z_block: pd.DataFrame,
                 weights_s: pd.Series,
                 k_max: int,
                 mass_cover: float,
                 err_budget: float,
                 sim_floor: float = PROTOTYPE_SIM_FLOOR,
                 seed_codes: list[int] | None = None,
                 err_mode: str = PROTOTYPE_ERR_MODE) -> tuple[list,int,float,float]:
    """
    Try K=1..k_max; return (prototypes, K, coverage, ERR).
    Coverage is fraction of weight whose assignment sim >= sim_floor.
    seed_codes is accepted but not used.
    """
    idx = Z_block.index
    w = weights_s.reindex(idx).fillna(0.0).values
    wtot = w.sum() + 1e-12

    best = None
    for K in range(1, min(k_max, len(idx)) + 1):
        prot = _facility_location(Z_block, K, w)
        assigned_codes, sim = _assign_hard_codes(Z_block, prot)        # labels + max sim
        # coverage by assignment at similarity floor
        cov = float(np.sum(w[sim >= sim_floor]) / wtot)
        err = _compute_err(Z_block, assigned_codes, sim, w, mode=err_mode)
        if (cov >= mass_cover) and (err <= err_budget):
            best = (prot, K, cov, err)
            break

    if best is None:
        K = min(k_max, len(idx))
        prot = _facility_location(Z_block, K, w)
        assigned_codes, sim = _assign_hard_codes(Z_block, prot)
        cov = float(np.sum(w[sim >= sim_floor]) / wtot)
        err = _compute_err(Z_block, assigned_codes, sim, w, mode=err_mode)
        best = (prot, len(prot), cov, err)

    return best


def _detect_nutrient_columns(raw_df: pd.DataFrame, preprocess_log=None) -> list[str]:
    """
    Return the list of nutrient feature columns from USDA df:
      - normalize names (replace '\n' with ' ')
      - apply NUTRIENT_NAME_MAPPING
      - keep numeric columns only
      - exclude non-nutrient meta fields
    """
    df = raw_df.copy()
    df.columns = df.columns.str.replace('\n', ' ', regex=False)
    df.rename(columns=NUTRIENT_NAME_MAPPING, inplace=True)

    # fields to never treat as nutrients (all are numeric strings in USDA so we must exclude)
    EXCLUDE = {
        'Food code',
        'Main food description',
        'WWEIA Category number',
        'WWEIA Category code',
        'WWEIA Category description',
    }

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    nutrients = [c for c in numeric_cols if c not in EXCLUDE]

    nutrients = sorted(nutrients)
    if preprocess_log is not None:
        preprocess_log.info(f"[NUTRIENTS] candidate numeric nutrient columns (post-rename) = {len(nutrients)}")
        for c in nutrients:
            preprocess_log.info(f"  - {c}")
    return nutrients


def _validate_nutrient_features(df: pd.DataFrame, features: list[str], preprocess_log=None) -> list[str]:
    """
    Ensure features exist, drop zero-variance cols, and log what we keep/drop.
    The df should already have normalized + renamed columns.
    """
    orig = list(features)
    # intersect with available columns
    features = [c for c in features if c in df.columns]
    missing = [c for c in orig if c not in df.columns]

    if preprocess_log is not None and missing:
        preprocess_log.warning(f"[NUTRIENTS] {len(missing)} requested features not in dataframe; they will be ignored.")
        for c in missing[:20]:
            preprocess_log.warning(f"  (missing) {c}")
        if len(missing) > 20:
            preprocess_log.warning(f"  ... and {len(missing)-20} more")

    # drop zero-variance columns (constant across foods)
    if features:
        X = df[features].astype(float)
        var = X.var(axis=0)
        keep = var[var > 0].index.tolist()
        dropped_const = [c for c in features if c not in keep]
        if preprocess_log is not None and dropped_const:
            preprocess_log.warning(f"[NUTRIENTS] dropping {len(dropped_const)} constant columns:")
            for c in dropped_const[:20]:
                preprocess_log.warning(f"  (const) {c}")
            if len(dropped_const) > 20:
                preprocess_log.warning(f"  ... and {len(dropped_const)-20} more")
        features = keep

    if preprocess_log is not None:
        preprocess_log.info(f"[NUTRIENTS] final active feature count = {len(features)}")

    return features


def build_systematic_prototype_mapping(preprocess_log) -> dict[int,int]:
    """
    Build {leaf_food_code -> prototype_food_code} with:
      - per-subcategory facility-location,
      - dynamic K allocation by mass/presence/diversity,
      - automatic K search s.t. coverage ≥ γ and error ≤ ε,
      - roll up tiny subcats (and foods below the similarity floor) to Main,
      - hard assignment; enforce a cosine similarity floor,
      - guarantee mapping for all observed meal codes,
      - keep a food as its own prototype if the prototype is not in the nutrient map.
    Top-presence 'seed' foods are computed per block but _auto_pick_K ignores them.
    Writes PROTOTYPE_MAPPING_PATH (data/mappings/food_to_prototype.pkl).
    """
    preprocess_log.info("Building nutrient-aware prototype mapping...")
    rows = pd.read_pickle('data/processed/real_meals/dataframes/combined_food_codes_standardized.pkl')

    with open(FOOD_CATEGORY_MAP_PATH, 'rb') as f:
        cat_map = pickle.load(f)
    nut_map = pd.read_pickle(FOOD_NUTRIENT_MAP_PATH)

    Z = _prep_nutrient_matrix(nut_map, cat_map)
    preprocess_log.info(f"[NUTRIENTS] ACTIVE_NUTRIENTS={len(ACTIVE_NUTRIENTS)} features")
    if len(ACTIVE_NUTRIENTS) < 8:
        preprocess_log.warning("[NUTRIENTS] very few features; check column detection / mapping.")


    weights = _food_weights_from_rows(rows, alpha=0.7) 

    # ---- dynamic K allocation across Subcats
    K_sub_alloc = allocate_K_per_subcat(cat_map, rows, Z)

    # presence vector to pick seeds per block
    presence = (rows[['SEQN','DAY','IFDCD']]
                .drop_duplicates()
                .groupby('IFDCD')
                .size()
                .astype(float))
    presence.index = presence.index.astype(int)

    mapping: dict[int,int] = {}
    deferred_idx: list[int] = []

    # ---- PASS 1: Subcategory
    for subcat, Zb in Z.groupby('Subcat', dropna=False):
        idx = list(Zb.index)
        if len(idx) < PROTOTYPE_MIN_SUBCAT:
            deferred_idx.extend(idx); continue

        # get dynamic K or fall back to old cap
        Kcap = K_sub_alloc.get(subcat, max(1, min(PROTOTYPE_K_MAX, int(np.ceil(PROTOTYPE_ALPHA * len(idx))))))

        # seeds: top presence foods in this block
        seeds = []
        if FORCE_PRESENCE_SEEDS > 0:
            seeds = (presence.reindex(idx).fillna(0.0)
                     .nlargest(min(FORCE_PRESENCE_SEEDS, len(idx)))
                     .index.tolist())

        prot, K, cov, err = _auto_pick_K(
            Zb, weights, Kcap, PROTOTYPE_MASS_COVER, PROTOTYPE_ERR_BUDGET,
            sim_floor=PROTOTYPE_SIM_FLOOR,
            seed_codes=seeds,
            err_mode=PROTOTYPE_ERR_MODE
        )
        preprocess_log.info(f"Subcat [{subcat}] n={len(idx)} -> K={K}, cov={cov:.2f}, ERR={err:.3f}")

        assigned_codes, sim = _assign_hard_codes(Zb, prot)
        for food, proto_code, s in zip(Zb.index, assigned_codes, sim):
            if s >= PROTOTYPE_SIM_FLOOR:
                mapping[int(food)] = int(proto_code)
            else:
                deferred_idx.append(int(food))

    # ---- PASS 2: roll-up leftovers at Main
    if deferred_idx:
        Z_rest = Z.loc[deferred_idx]
        for main, Zm in Z_rest.groupby('Main'):
            idx = list(Zm.index)
            # simple fallback cap at Main
            Kcap = max(1, min(PROTOTYPE_K_MAX, int(np.ceil(PROTOTYPE_ALPHA * len(idx)))))
            # seeds at Main by presence
            seeds = []
            if FORCE_PRESENCE_SEEDS > 0:
                seeds = (presence.reindex(idx).fillna(0.0)
                         .nlargest(min(FORCE_PRESENCE_SEEDS, len(idx)))
                         .index.tolist())
            prot, K, cov, err = _auto_pick_K(
                Zm, weights, Kcap, PROTOTYPE_MASS_COVER, PROTOTYPE_ERR_BUDGET,
                sim_floor=PROTOTYPE_SIM_FLOOR,
                seed_codes=seeds,
                err_mode=PROTOTYPE_ERR_MODE
            )
            preprocess_log.info(f"Main [{main}] n={len(idx)} -> K={K}, cov={cov:.2f}, ERR={err:.3f}")
            assigned_codes, sim = _assign_hard_codes(Zm, prot)
            for food, proto_code, s in zip(Zm.index, assigned_codes, sim):
                mapping[int(food)] = int(proto_code) if s >= PROTOTYPE_SIM_FLOOR else int(food)

    # ---- guarantee mapping for every observed code in meals
    meal_codes = rows['IFDCD'].astype(int).unique().tolist()
    for c in meal_codes:
        mapping.setdefault(int(c), int(c))

    # ---- constrain mapping to codes that exist in nutrient map to avoid introducing unknown IDs
    try:
        allowed = set(pd.to_numeric(nut_map['Food code'], errors='coerce').dropna().astype(int).tolist()) if 'Food code' in nut_map.columns else set()
        if allowed:
            replaced = 0
            for k, v in list(mapping.items()):
                if int(v) not in allowed and int(k) in allowed:
                    mapping[k] = int(k)
                    replaced += 1
            preprocess_log.info(f"[NUTRIENTS] constrained prototype mapping: replaced {replaced} targets not in nutrient map")
        else:
            preprocess_log.warning("[NUTRIENTS] could not derive allowed nutrient codes; skipping constraint")
    except Exception as e:
        preprocess_log.warning(f"[NUTRIENTS] error constraining prototype mapping: {e}")

    os.makedirs(os.path.dirname(PROTOTYPE_MAPPING_PATH), exist_ok=True)
    with open(PROTOTYPE_MAPPING_PATH, 'wb') as f:
        pickle.dump(mapping, f)

    preprocess_log.info(f"Saved prototype mapping: {PROTOTYPE_MAPPING_PATH} (size={len(mapping)})")
    preprocess_log.info(f"Prototype mapping size: {len(mapping)}")
    self_frac = np.mean([k == v for k, v in mapping.items()])
    preprocess_log.info(f"Self-mapped fraction: {self_frac:.3f}")
    return mapping


def _apply_prototype_mapping_to_meal(meal, mapping: dict[int,int], allowed_codes: set[int] | None = None) -> dict[int,float]:
    """Aggregate a Meal’s leaf foods to prototype grams (mass-conserving)."""
    agg = {}
    for code, grams in zip(meal.food_codes, meal.gram_portions):
        try:
            code_int = int(code)
        except Exception:
            code_int = int(float(code))
        # Enforce allowed set on source and target
        if allowed_codes is not None and code_int not in allowed_codes:
            # skip foods with no nutrient/category mapping
            continue
        proto = mapping.get(code_int, code_int)
        if allowed_codes is not None and proto not in allowed_codes:
            # fallback to self if prototype not allowed
            proto = code_int
        agg[proto] = agg.get(proto, 0.0) + float(grams)
    return agg


def standardize_food_codes(preprocess_log):
    """Load (or build) the discontinued-code map and the standardized meals: writes
    data/mappings/discontinued_food_codes.pkl, combined_food_codes_standardized.pkl and
    <meal>_meals.pkl unless they already exist."""
    # Note: the last two entries both name .../2017/DataFiles/DR1IFF_H.xpt. Files are cached by
    # name in data/raw/meals/, so 2013-2014 Day 1 is read three times and no 2017-2020 file is
    # read. Kept as is: the paper's dataset was built from this list.
    meal_survey_urls = [
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DR1IFF_H.xpt", # 2013-2014 Day 1
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DR2IFF_H.xpt", # 2013-2014 Day 2
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR1IFF_I.xpt", # 2015-2016 Day 1
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DR2IFF_I.xpt", # 2015-2016 Day 2
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DR1IFF_H.xpt", # meant as 2017-2020 Day 1 (see note)
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DR1IFF_H.xpt", # meant as 2017-2020 Day 2 (see note)
    ]
    discontinued_urls = [
        "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/Discontinued_Codes_between_FNDDS_2011-2012_and_FNDDS_2013-2014.xlsx",              # 2013-2014
        "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/Discontinued_Food_Codes_between_FNDDS_2013-2014_and_FNDDS_2015-2016_7_19_18.xlsx", # 2015-2016
        "https://www.ars.usda.gov/ARSUserFiles/80400530/apps/Discontinued_Food_Codes_between_FNDDS_2017-2018_and_FNDDS_2019-2020_013123.xlsx",  # 2017-2020
    ]
        
    cache_dir = "data/raw/meals/"
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists('data/mappings/discontinued_food_codes.pkl'):
        discontinued_id_map = generate_discontinued_id_map(discontinued_urls, cache_dir+'/discontinued', preprocess_log)
    else:
        preprocess_log.info("File exists, loading discontinued ID mapping")
        discontinued_id_map = pd.read_pickle('data/mappings/discontinued_food_codes.pkl')    

    if not os.path.exists('data/processed/real_meals/dataframes/combined_food_codes_standardized.pkl'):
        combined_meal_data, breakfast_meals, lunch_meals, dinner_meals = load_and_process_meal_data(meal_survey_urls, discontinued_id_map, cache_dir, preprocess_log)
    else:
        combined_meal_data  = pd.read_pickle("data/processed/real_meals/dataframes/combined_food_codes_standardized.pkl")
        breakfast_meals     = pd.read_pickle("data/processed/real_meals/breakfast_meals.pkl")
        lunch_meals         = pd.read_pickle("data/processed/real_meals/lunch_meals.pkl")
        dinner_meals        = pd.read_pickle("data/processed/real_meals/dinner_meals.pkl")   

    preprocess_log.info(f"Processed a total of {len(breakfast_meals)} breakfast meals, {len(lunch_meals)} lunch meals, and {len(dinner_meals)} dinner meals")
    unique_food_codes = combined_meal_data['IFDCD'].nunique()
    preprocess_log.info(f"Processed unique food codes: {unique_food_codes}")


def normalize_macronutrients(macronutrients):
    """
    Normalize the macronutrient names to ensure they match the column names in the DataFrame.

    Args:
        macronutrients (list): List of macronutrient names to normalize.
    
    Returns:
        list: Normalized macronutrient names.
    """
    return [name.strip().replace('\n', ' ').lower() for name in macronutrients]

def nutrient_mapping_and_conversions(aggregated_nutrition_per_day):
    """Replace newlines in the column names and rename them with NUTRIENT_NAME_MAPPING
    (in place). Despite the name, no unit conversion is applied."""
    aggregated_nutrition_per_day.columns = aggregated_nutrition_per_day.columns.str.replace('\n', ' ')
    aggregated_nutrition_per_day.rename(columns=NUTRIENT_NAME_MAPPING, inplace=True)
    return aggregated_nutrition_per_day


def calculate_average_macronutrients(meal_data, food_nutrient_mapping, macronutrients):
    """
    Mean of each macronutrient column over the distinct foods that occur in meal_data
    (one value per food; not weighted by meals or grams).

    Args:
        meal_data (pd.DataFrame): DataFrame containing food codes as columns.
        food_nutrient_mapping (pd.DataFrame): DataFrame containing the mapping of food codes to nutrients.
        macronutrients (list): List of macronutrients to focus on.

    Returns:
        avg_macronutrients (dict): Dictionary with macronutrient averages for the dataset.
    """       
    # Get food codes with non-zero values in the meal_data
    non_zero_columns = meal_data.columns[(meal_data != 0).any()].tolist()

    # Filter out the rows from the food_nutrient_mapping corresponding to the food codes in the meal_data
    nutrient_data = food_nutrient_mapping[food_nutrient_mapping['Food code'].isin(non_zero_columns)]
    # Ensure only the required macronutrient columns are selected
    nutrient_data = nutrient_data[[col for col in macronutrients if col in nutrient_data.columns]]


    # Calculate the average of the macronutrient values for the meals
    avg_macronutrients = nutrient_data.mean()

    # Return the average as a dictionary
    return avg_macronutrients.to_dict()

def plot_spider_chart(rdi, macronutrients, avg_macronutrients, meal_types):
    """
    Radar chart of each meal type's average macronutrients as a fraction of the daily
    targets; saved as data/processed/real_meals/diagnostics/nutrient_target_rdi.svg.

    Args:
        rdi (pd.DataFrame): One-row frame of daily targets, columns named as in MACRONUTRIENTS.
        macronutrients: only its length is used (the axes iterate over MACRONUTRIENTS).
        avg_macronutrients (dict): Dictionary with average macronutrients for each meal type.
        meal_types (list): not used.
    """
    # Normalize macronutrient names
    macronutrients = normalize_macronutrients(macronutrients)
    
    num_vars = len(macronutrients)  # Number of variables (macronutrients)

    # Create a list of angles for the spider chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Repeat the first angle to close the plot (full circle)
    angles += angles[:1]

    # Create figure and axis for the spider plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Daily targets for the macronutrient axes (the targets themselves are not drawn)
    rdi = rdi[[col for col in MACRONUTRIENTS if col in rdi.columns]]
    rdi_values = rdi.values.tolist()[0]

    colors = plt.get_cmap("Paired")
    # Plot the average macronutrients for each meal type
    for i, (meal_type, avg_macronutrient) in enumerate(avg_macronutrients.items()):
        values = [(avg_macronutrient.get(nutrient, 0) / rdi[nutrient]) for nutrient in MACRONUTRIENTS]
        values += values[:1]
        # Paired colormap: blue for breakfast, green for lunch, red for dinner
        if meal_type == 'breakfast':
            color = colors(1)  # Blue
        elif meal_type == 'lunch':
            color = colors(3)  # Green
        else:
            color = colors(5)  # Red
        ax.plot(angles, values, label=f'{meal_type.capitalize()}', color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.3)

    # Add labels and title
    ax.set_title('Comparison of Average Macronutrients with RDI', size=16, color='black', y=1.1)
    
    # Add labels to each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(MACRONUTRIENTS, size=14)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.04, 0.08, 0.12, 0.16, 0.2])  # radial ticks: fraction of the daily target
    ax.set_yticklabels(['0.04', '0.08', '0.12', '0.16', '0.20'], fontsize=12, color='black')

    ax.grid(True)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=12)
    
    plt.tight_layout()
    out_svg = 'data/processed/real_meals/diagnostics/nutrient_target_rdi.svg'
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    plt.savefig(out_svg, format='svg')
    plt.close(fig)

def main():
    preprocess_log = setup_logger('preprocess_log', 'logs/preprocessing/preprocessing.log')
    preprocess_log.info("Creating dataset...")
    standardize_food_codes(preprocess_log)
    
    # Build the nutrient-aware prototype mapping once (from FOOD_CATEGORY_MAP_PATH and
    # FOOD_NUTRIENT_MAP_PATH) and log how it changes the meal food codes
    if not os.path.exists(PROTOTYPE_MAPPING_PATH):
        build_systematic_prototype_mapping(preprocess_log)
        with open(PROTOTYPE_MAPPING_PATH, 'rb') as f:
            mapping = pickle.load(f)

        # 1) Universe stats
        total_codes_in_mapping = len(mapping)
        self_map_fraction = np.mean([k == v for k, v in mapping.items()])
        preprocess_log.info(f"[CHECK] mapping size={total_codes_in_mapping}, self-map frac={self_map_fraction:.3f}")

        # 2) Limit to codes that actually occur in the meals
        rows = pd.read_pickle('data/processed/real_meals/dataframes/combined_food_codes_standardized.pkl')
        meal_codes = rows['IFDCD'].astype(int).unique().tolist()
        meal_codes_set = set(meal_codes)

        covered = {k: v for k, v in mapping.items() if k in meal_codes_set}
        uncovered = [c for c in meal_codes if c not in covered]

        changed = sum(1 for k, v in covered.items() if k != v)
        unchanged = len(covered) - changed
        unique_prototypes_used = len(set(covered.values()))

        preprocess_log.info(f"[CHECK] meal codes present={len(meal_codes)}")
        preprocess_log.info(f"[CHECK] meal codes with a mapping={len(covered)} (missing in map={len(uncovered)})")
        preprocess_log.info(f"[CHECK] of mapped meal codes: changed={changed} (to a different proto), unchanged={unchanged}")
        preprocess_log.info(f"[CHECK] unique prototypes actually used by meals={unique_prototypes_used}")
        preprocess_log.info(f"[CHECK] column reduction factor (distinct foods -> distinct prototypes)={len(covered)/max(1,unique_prototypes_used):.2f}x")

        # 3) Gram-weighted view: how much mass moved to a different code?
        wtcol = 'WTD1' if 'WTD1' in rows.columns else ('WTD2' if 'WTD2' in rows.columns else None)
        if wtcol:
            mass = (rows['IGRMS'] * rows[wtcol].fillna(1.0))
        else:
            mass = rows['IGRMS']

        mass_by_code = mass.groupby(rows['IFDCD'].astype(int)).sum()
        mass_total = float(mass_by_code.sum())
        mass_changed = float(mass_by_code[[k for k, v in covered.items() if k != v]].sum())
        mass_self     = float(mass_by_code[[k for k, v in covered.items() if k == v]].sum())

        preprocess_log.info(f"[CHECK] mass total={mass_total:,.0f} g, mass changed={mass_changed/mass_total:.2%}, mass self={mass_self/mass_total:.2%}")

        # 4) (optional) Top 15 prototypes by accumulated mass after roll-up
        proto_mass = {}
        for k, grams in mass_by_code.items():
            proto = covered.get(int(k), int(k))
            proto_mass[proto] = proto_mass.get(proto, 0.0) + float(grams)

        top15 = sorted(proto_mass.items(), key=lambda kv: kv[1], reverse=True)[:15]
        preprocess_log.info("[CHECK] top 15 prototypes by accumulated grams after roll-up:")
        for p, g in top15:
            preprocess_log.info(f"  proto={p}: {g:,.0f} g")
            

    meal_types = ['breakfast', 'lunch', 'dinner']
    # Final order: raw -> standardized -> prototype_mapped -> outliers_removed -> bci_filtered
    processing_types = ['raw', 'food_codes_standardized', 'prototype_mapped', 'outliers_removed', 'bci_filtered']
    for meal_type in meal_types:
        if not os.path.exists(f'data/processed/real_meals/dataframes/{meal_type}_food_codes_standardized.pkl') or not os.path.exists(f'data/processed/real_meals/dataframes/{meal_type}_prototype_mapped.pkl'):
            create_meal_df(meal_type, preprocess_log)
        preprocessing(meal_type, preprocess_log)
    preprocessing_summary(meal_types, processing_types, preprocess_log)
        
    
    breakfast_data_path = 'data/processed/real_meals/dataframes/breakfast_sparsity_filtered.pkl'
    lunch_data_path = 'data/processed/real_meals/dataframes/lunch_sparsity_filtered.pkl'
    dinner_data_path = 'data/processed/real_meals/dataframes/dinner_sparsity_filtered.pkl'
    food_nutrient_mapping_path = 'data/mappings/food_to_nutrient.pkl'

    food_nutrient_mapping = pd.read_pickle(food_nutrient_mapping_path)

    # Load data for breakfast, lunch, and dinner
    breakfast_data = pd.read_pickle(breakfast_data_path)
    lunch_data = pd.read_pickle(lunch_data_path)
    dinner_data = pd.read_pickle(dinner_data_path)

    food_nutrient_mapping = nutrient_mapping_and_conversions(food_nutrient_mapping)

    # Calculate average macronutrients for each meal type
    avg_breakfast = calculate_average_macronutrients(breakfast_data, food_nutrient_mapping, MACRONUTRIENTS)
    avg_lunch = calculate_average_macronutrients(lunch_data, food_nutrient_mapping, MACRONUTRIENTS)
    avg_dinner = calculate_average_macronutrients(dinner_data, food_nutrient_mapping, MACRONUTRIENTS)

    # Combine the average macronutrients for all meal types
    avg_macronutrients = {
        'breakfast': avg_breakfast,
        'lunch': avg_lunch,
        'dinner': avg_dinner
    }
    
    calorie_level = '2000 cal'
    nutrient_targets = load_nutrient_targets(calorie_level)
    nutrient_targets = nutrient_targets.to_frame().transpose().reset_index()
    nutrient_targets = nutrient_mapping_and_conversions(nutrient_targets)
    print(nutrient_targets)
    # Plot the spider chart
    plot_spider_chart(nutrient_targets, MACRONUTRIENTS, avg_macronutrients, ['breakfast', 'lunch', 'dinner'])


def load_nutrient_targets(calorie_level='2000 cal'):
    # Load the nutrient targets from the CSV file
    nutrient_targets_df = pd.read_csv('data/raw/food_patterns/healthy_US_style_food_pattern.csv', index_col=0)
    # Select the column corresponding to the specified calorie level
    nutrient_targets = nutrient_targets_df[calorie_level]
    return nutrient_targets


if __name__ == "__main__":
    main()