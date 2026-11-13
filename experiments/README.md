# Substitution analysis (Figure 4)

How few food changes turn a real meal into a healthier, cheaper one. For every generated meal, the
analysis looks for real meals that differ from it by exactly k foods (k = 1, 2, 3); the generated
meal is then a k-food substitute for that real meal. Each candidate is scored on nutrition gain and
cost change, a trade-off parameter θ weights the two, and the best candidate per generated meal is
kept. Sweeping θ traces a cost–benefit curve for each k, and its knee is the operating point shown
in Figure 4.

| File | Make targets | Role |
|---|---|---|
| [`analyze_meal_substitutions.py`](analyze_meal_substitutions.py) | `substitutions`, `theta_sweep`, `sup-substitutions` | Finds and scores substitutions for one θ and k |
| [`plot_substitution_figures.py`](plot_substitution_figures.py) | `substitutions`, `theta_sweep` | Category heatmaps per run; knee θ per k, Figure 4B and Figure 4C–D across the sweep |
| [`substitution_foldchange/`](substitution_foldchange/) | `figure4a` | Figure 4A, a worked example ([README](substitution_foldchange/README.md)) |
| [`costs/restaurant_pricing.py`](costs/restaurant_pricing.py) | (imported) | Portion-based restaurant prices (S1 Table 16) |

All inputs come from `make eval` (`outputs/runs/<RUN_NAME>/data_exports/`); outputs go to
`outputs/runs/<RUN_NAME>/substitutions_k3/`.

## `analyze_meal_substitutions.py`

For each meal type:

1. **Candidates.** For each generated meal, the real meals of the same meal type whose energy is
   within ±5% (`--kcal_caliper`) and whose number of foods is within ±1 (`--items_caliper`), and
   that differ by exactly k foods (`--require_k_substitution k`; k = the larger of foods added and
   foods removed). The `--k_nearest` most similar candidates are kept (a blend of cosine similarity
   of the gram vectors and Jaccard overlap of the foods).
2. **Nutrition gain (ΔRDI).** The reduction in the meal's mean absolute deviation from its
   nutrient targets, |intake / target − 1| averaged over nutrients, when the real meal is replaced by
   the generated one. Targets are the daily RDI times the meal's share of daily energy (25%, 35% and
   40% for breakfast, lunch and dinner).
3. **Cost change (ΔCost).** With `--use_restaurant_portion_pricing`, both meals are priced with the
   portion-based restaurant model plus a flat `--restaurant_overhead_usd`; otherwise from the
   per-100 g price table (`data/prices/price_map.csv`) with built-in food-category multipliers.
   `--no_cost_increase` drops substitutes that cost more.
4. **Score.** `w × ΔRDI% + (1 − w) × saving% − w × cost increase%` with `w = θ / (1 + θ)`: θ = 0
   scores savings only, larger θ favours nutrition. Candidates that worsen nutrition or score below
   zero are dropped. Category rules can exclude adding or removing beverages and "other" foods and
   penalise adding mixed dishes, more strongly for larger k (`--mixed_cross_alpha`,
   `--mixed_cross_beta`).
5. **Winner.** The best-scoring candidate per generated meal.

`make substitutions` and `make theta_sweep` call it once per θ and k with the settings in the
Makefile's `SUB_ARGS` (50 nearest meals, restaurant pricing with a $2 overhead, no cost increases,
the category rules above). `make sup-substitutions` uses the script's defaults instead (three
nearest meals, θ = 0.5) for the S1 cost tables and the substitution counts.

Matching runs over all real meals of the meal type.

Outputs in `--output_dir`:

| File | Contents |
|---|---|
| `substitution_matches.csv` | One row per selected substitution: meal type, `gen_idx`, `real_idx`, `k_substitution`, nutrition gain (`improvement`, `dRDI_pct`), costs (`real_cost`, `gen_cost`, `delta_cost`, `dCost_pct`), changes in MAR, AMDR, energy density, protein, fibre and sodium, and the foods and food categories added and removed |
| `substitution_stats_overall.csv`, `substitution_stats_by_k*.csv` | Medians of the gain and cost change, share of substitutions that save money |
| `bootstrap_results.csv` | Bootstrap CIs of the main statistics per meal type and k |
| `substitution_summary.csv` | Counts and mean improvements per meal type |
| `substitution_examples_detailed.csv/.txt` | The best examples, written out food by food |

`dCost_pct` is relative to the meal cost without the overhead; `real_cost` and `gen_cost` include it.

## `plot_substitution_figures.py`

- `--mode k_from_input` (`make substitutions`): for one run, a heatmap of which food category is
  removed for which category added, and the matching add-versus-remove scatter; with runs for
  k = 1, 2 and 3 side by side, also a combined panel.
- `--mode optimal_theta` (`make theta_sweep`): groups the runs of each θ, computes for every θ and k
  the median nutrition gain and cost saving of the selected substitutions (with bootstrap CIs), and
  picks for each k the knee of that curve: the θ whose point lies farthest from the straight line
  joining the curve's two ends. It writes `optimal_theta_by_k.json`/`.txt`, Figure 4B
  (`figure_cost_benefit_points_by_theta_and_k`), Figure 4C–D at the knee θ values
  (`figure_category_scatter_plus_heatmap_k123`) and, with `--dump_debug`, the per-θ points as CSV
  (`debug_points_by_theta_and_k_theta_*.csv`, read by `scripts/generate_manuscript_numbers_log.py`).

The legend counts in Figure 4B add up the selected substitutions of every θ run, so each generated
meal is counted once per θ. Ternary plots (`--with_ternary`) are optional extras and are not part of
the paper.

## `costs/restaurant_pricing.py`

Prices meals the way a restaurant menu would: each food is mapped by name to a menu item with a
portion size (grams) and a price per portion, the number of portions is the meal's grams divided by
the portion size, capped per item, and the meal cost is the sum over its foods. The pairs are those
listed in S1 Table 16. Matching is by keywords in the food name, so it depends on the FNDDS
descriptions; for example, any food whose name contains "water" is priced as water ($0).
`estimate_restaurant_cost` prices one meal vector and `estimate_restaurant_costs_for_meals` a
matrix of meals.
