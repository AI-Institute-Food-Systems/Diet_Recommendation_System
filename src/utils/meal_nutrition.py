"""Meal-level nutrient totals for the clustering features (src/utils/clustering_engine.py, `make clustering`)."""

# FNDDS column names -> names used in the RDI table. Renames only: values keep their FNDDS
# units (e.g. vitamin D stays in mcg). Same mapping as src/preprocessing/data_preprocessing.py.
NUTRIENT_NAME_MAPPING = {
    "Fiber, total dietary (g)": "Fiber (total dietary) (g)",
    "Fatty acids, total saturated (g)": "Saturated fats (g)",
    "Fatty acids, total monounsaturated (g)": "Monounsaturated fats (g)",
    "Fatty acids, total polyunsaturated (g)": "Polyunsaturated fats (g)",
    "20:5 n-3 (g)": "EPA (20:5 n-3) (g)",
    "22:6 n-3 (g)": "DHA (22:6 n-3) (g)",
    "18:0 (g)": "Stearic Acid (18:0) (g)",
    "Choline, total (mg)": "Choline (mg)",
    "Vitamin A, RAE (mcg_RAE)": "Vitamin A (mg_RAE)",
    "Vitamin D (D2 + D3) (mcg)": "Vitamin D (IU)",
    "Vitamin K (phylloquinone) (mcg)": "Vitamin K (mg)",
    "Folate, total (mcg)": "Folate (mg_DFE)",
}


def calculate_meal_nutrition_from_df(meal_df, nutrient_data):
    """Nutrient totals of one meal.

    meal_df: grams per food, a Series indexed by food code. nutrient_data: nutrients per 100 g
    indexed by food code (utils.helpers.load_nutrient_data). Returns a Series of totals with
    newlines removed from the nutrient names and NUTRIENT_NAME_MAPPING applied.
    """
    meal_df = meal_df[meal_df > 0].reset_index()
    meal_df.columns = ["food_code", "portion"]
    meal_df["food_code"] = meal_df["food_code"].astype(int)

    relevant_nutrients = nutrient_data.loc[nutrient_data.index.intersection(meal_df["food_code"])]
    merged_data = meal_df.merge(relevant_nutrients, left_on="food_code", right_index=True)
    nutrient_columns = [col for col in merged_data.columns if col not in ["food_code", "portion"]]
    scaled_nutrients = merged_data[nutrient_columns].div(100).mul(merged_data["portion"], axis=0)
    meal_nutrients = scaled_nutrients.sum()
    meal_nutrients.index = meal_nutrients.index.str.replace("\n", " ", regex=True)
    meal_nutrients.rename(index=NUTRIENT_NAME_MAPPING, inplace=True)
    return meal_nutrients
