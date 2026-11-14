# CVAE versus GPT meals

Compares the CVAE's meals with meals written by an OpenAI GPT model (GPT-4o by default) for the
same 34 archetypes and the same nutrient targets (Results, "comparison with large language
models"; S1 §B.3). It is optional and not part of `make reproduce`, because generating the GPT
meals needs an OpenAI API key and is billed by OpenAI.

| File | Make target | Role |
|---|---|---|
| [`generate_meals.py`](generate_meals.py) | `gpt-generate` | CVAE and GPT meals for every archetype (calls the OpenAI API) |
| [`evaluate.py`](evaluate.py) | `gpt-evaluate` | Nutrition metrics of both sets of meals (no API access) |
| [`common.py`](common.py) | (imported) | Run inputs, the CVAE generator, allowed foods, per-meal targets |

## Running it

After `make reproduce` (or at least `make cvae_train eval`):

```bash
pip install -r requirements-gpt.txt      # the openai package
cp .env.example .env                     # then put your key after OPENAI_API_KEY=
make gpt-comparison                      # = make gpt-generate gpt-evaluate
```

`OPENAI_API_KEY` can also be exported in the shell instead of using `.env`; a variable that is
already set takes precedence. `.env` is git-ignored. With the defaults, `make gpt-generate` makes
about 340 API calls (34 archetypes × 100 meals in batches of 10).

| Variable | Default | Meaning |
|---|---|---|
| `GPT_MODEL` | `gpt-4o` | OpenAI chat model |
| `GPT_N` | `100` | Meals per archetype and method |
| `GPT_BATCH` | `10` | Meals requested per API call |
| `GPT_OUT` | `outputs/runs/<RUN_NAME>/gpt_comparison/<GPT_MODEL>` | Output folder |
| `GPT_RESULTS` | `$(GPT_OUT)/experiment_results.json` | Input of `make gpt-evaluate` |

For a quick, cheap test: `python experiments/gpt_comparison/generate_meals.py --max_archetypes 2
--n_per_archetype 10`.

## `generate_meals.py`

For each (meal type, archetype) pair the model was trained on:

1. **Allowed foods.** The foods the CVAE's decoder gate allows for the archetype (and that have
   nutrient data), so both methods choose from the same foods.
2. **CVAE meals.** `GPT_N` meals from the trained model with the settings of `make eval`
   (presence sampling, then RDI-per-kcal portioning toward the meal's targets), seeded with
   `SEED + pair code`.
3. **GPT meals.** The model is asked for `GPT_N` meals in batches of `GPT_BATCH`. The prompt gives
   the meal type, the targets (energy 500 / 700 / 800 kcal ±5% for breakfast / lunch / dinner,
   protein ≥ 20 / 25 / 28 g, fibre ≥ 7.5 / 8.5 / 9.5 g, sodium ≤ 450 / 700 / 800 mg, added sugars
   and saturated fat ≤ 10% of energy, beverage energy ≤ 25%), the structure (2–4 items, at most one
   beverage, at least 2 / 3 / 3 solid items, solids ≤ 300 g, beverages ≤ 500 / 400 / 400 g), typical
   portion ranges, and the allowed foods as FNDDS code → name. The answer must follow a strict JSON
   schema (meals of 2–4 items, each an 8-digit code and grams). Items with codes outside the list
   or grams outside (0, 1000] are dropped; a meal left empty, or missing from the answer, counts as
   failed. Rate-limited calls are retried with backoff, answers cut off by the length limit are
   split into smaller batches, and a batch with no usable meal is asked again up to three times.

Sampling uses `--temperature` and `--top_p` (both 1.0 by default); reasoning models (o-series,
GPT-5) are called with low reasoning effort and no sampling parameters.

Output in `GPT_OUT`: `experiment_results.json` with the settings, every CVAE and GPT meal (pair,
meal type, archetype, sample number, success, grams per food code, error) and the raw GPT answers;
`generate.log`.

## `evaluate.py`

Builds a gram matrix per method and meal type from the successful meals, looking each food up by
its FNDDS code in the run's foods (`data_exports/` from `make eval`), and computes with
`src/analysis_rdi_eval/nutrition_metrics.py`:

- energy (kcal) and the protein, fat and carbohydrate shares of energy;
- AMDR compliance: protein 10–35%, fat 20–35% and carbohydrate 45–65% of energy, all at once;
- MAR (mean adequacy ratio over 11 vitamins and minerals, each capped at 1) against the daily RDI
  times the meal's share of daily energy (25 / 35 / 40%);
- MER with the moderation limits of Figure 3 (sodium, saturated fat, total sugars);
- diversity: evenness of the gram amounts across the meal's foods (Shannon entropy divided by
  its maximum, log of the number of foods; foods under 5 g are ignored).

Output in `<results folder>/evaluation/`:

| File | Contents |
|---|---|
| `per_meal_metrics_<cvae\|gpt>_<meal type>.csv` | One row per meal: archetype, energy, macronutrient shares, AMDR compliance, MAR, MER, diversity |
| `summary_metrics.csv` | Mean, SD and median of each metric per method and meal type |
| `comparison.json`, `comparison.txt` | Per method: meals evaluated, meal-weighted AMDR compliance, macronutrient shares and diversity over all meal types |

`make gpt-evaluate GPT_RESULTS=<file>` evaluates any results file in this format, for example one
made with another model or another run.
