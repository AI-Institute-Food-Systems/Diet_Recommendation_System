# Data

Inputs to the pipeline. Everything in this folder is read-only for the pipeline except
`data/processed/` and `data/mappings/food_to_prototype.pkl`, which `make preprocess` writes
(both are git-ignored).

```text
data/
├── raw/
│   ├── meals/                  NHANES / WWEIA individual-foods files
│   │   ├── DR1IFF_H.xpt        2013–2014, day 1
│   │   ├── DR2IFF_H.xpt        2013–2014, day 2
│   │   ├── DR1IFF_I.xpt        2015–2016, day 1
│   │   └── DR2IFF_I.xpt        2015–2016, day 2
│   └── food_patterns/
│       └── healthy_US_style_food_pattern.csv
├── mappings/
│   ├── food_to_nutrient.csv / .pkl
│   ├── food_ingredient_map.csv / .pkl
│   └── discontinued_food_codes.pkl
└── prices/
    └── price_map.csv
```

## Sources

| File | Contents | Source |
|---|---|---|
| `raw/meals/DR?IFF_?.xpt` | One row per food eaten, with the eating occasion, food code and grams, for every respondent of the NHANES dietary recall (SAS transport format) | [NHANES](https://wwwn.cdc.gov/nchs/nhanes/), CDC (public domain) |
| `raw/food_patterns/healthy_US_style_food_pattern.csv` | Daily nutrient amounts of the Healthy US-Style Food Pattern at 1,000–3,200 kcal; the pipeline uses the 2,000 kcal column for the RDI targets | Dietary Guidelines for Americans |
| `mappings/food_to_nutrient.*` | Nutrients per 100 g and WWEIA category for 9,260 FNDDS food codes | FNDDS, USDA ARS |
| `mappings/food_ingredient_map.*` | Ingredients and their weights for each FNDDS food code, with WWEIA main and subcategory | FNDDS, USDA ARS |
| `mappings/discontinued_food_codes.pkl` | Food codes discontinued between FNDDS releases, mapped to their replacements (279 codes) | FNDDS change lists, USDA ARS |
| `prices/price_map.csv` | Price per 100 g for the foods the model uses, with a justification and a source for each price | Compiled for the paper from retail price data |

The `.csv` and `.pkl` versions of a mapping hold the same table; the code reads the `.pkl`.

The recall files are committed so that a clone runs without network access. If
`mappings/discontinued_food_codes.pkl` is ever missing, preprocessing rebuilds it from the USDA
change lists listed in `src/preprocessing/data_preprocessing.py`.

## Preprocessing outputs

`make preprocess` writes, per meal type, to `data/processed/real_meals/dataframes/`:

| File | Step |
|---|---|
| `combined_food_codes_standardized.pkl`, `<meal>_food_codes_standardized.pkl` | Meals built from the recall files (breakfast, lunch, dinner by eating occasion), with discontinued codes replaced |
| `<meal>_prototype_mapped.pkl` | Foods rolled up to prototype foods; the map is saved to `data/mappings/food_to_prototype.pkl` |
| `<meal>_outliers_removed.pkl` | Local Outlier Factor removes 0.3% of meals |
| `<meal>_bci_filtered.pkl` | Foods kept by a bootstrap confidence-interval test on their frequency; meals left without food are dropped. **Clustering and training read this file**: 39,435 breakfast, 37,024 lunch and 42,831 dinner meals over 528, 627 and 686 foods |
| `<meal>_sparsity_filtered.pkl` | An additional sparsity filter that is computed for diagnostics but not used downstream |

Each file is a meal × food table of grams. Diagnostic plots and metrics of the sparsity step go to
`data/processed/real_meals/diagnostics/`, and the meal and food counts after every step are written
to `logs/preprocessing/preprocessing.log`.
