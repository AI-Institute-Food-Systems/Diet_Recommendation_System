# Figure 4A: worked substitution example

[`plot_nutrient_fold_changes.py`](plot_nutrient_fold_changes.py) draws the worked example of
Figure 4A: a real meal, a one-food (1-hop) and a two-food (2-hop) substitute, their costs and
RDI improvement, and the per-nutrient ratio of each substitute to the original meal.

`make figure4a` runs it after `make theta_sweep`:

```bash
python -m experiments.substitution_foldchange.plot_nutrient_fold_changes \
  --sweep_dir outputs/runs/eval_run/substitutions_k3 \
  --data_dir outputs/runs/eval_run \
  --outdir outputs/runs/eval_run/substitutions_k3/figure4a
```

`--sweep_dir` reads the selected θ for k = 1 and k = 2 from
`figures_theta_sweep/optimal_theta_by_k.json` and loads the substitution matches at those θ values.
The script then picks a real meal that has both a 1-hop and a 2-hop substitute that is healthier
and not more expensive, ranks the candidates and draws the selected one.

## Outputs

| File | Contents |
|---|---|
| `fixed_meal_composition.svg/.png` | The original meal and its two substitutes, with cost and ΔRDI (Figure 4A) |
| `figure_substitution_foldchange_two_panels.svg/.png` | Ratio of each nutrient in the substitute to the original meal, log scale |
| `figure_meal_compositions.svg/.png` | The meal compositions on their own |
| `candidate_meals_top50.csv/.json` | Other candidate meals with both substitutes, for choosing a different example |

## Choosing the example

- `--base MealType:real_idx` draws the substitutes of one particular real meal.
- `--pair MealType:gen_idx,real_idx` (given twice, for k = 1 and k = 2) draws exactly those pairs.
- `--matches_csv PATH` (repeatable) uses other substitution runs instead of `--sweep_dir`.

The selected example depends on the generated meals of the run, so different models give
different examples.

## Notes

- Costs use the portion-based restaurant pricing in
  [`experiments/costs/restaurant_pricing.py`](../costs/restaurant_pricing.py) plus a $2.00 overhead;
  ΔRDI is the change in deviation from the per-meal RDI targets, in percent.
- For display, every column also lists a side salad, a fruit cup and iced tea. These items are
  not part of the computed meals, their costs or ΔRDI.
- Food names are shortened for display (for example, portion descriptions are rounded and some
  long FNDDS descriptions are replaced by common names).
