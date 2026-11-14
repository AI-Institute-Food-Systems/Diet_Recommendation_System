# Changelog

## v1.1.0 (2026-11-14)

- `make gpt-comparison`: the comparison with GPT-4o (Results; S1 §B.3). `make gpt-generate` asks
  an OpenAI model for meals from each archetype's foods with the CVAE's targets and generates the
  CVAE's meals for the same archetypes; `make gpt-evaluate` computes AMDR compliance,
  macronutrient shares, diversity, MAR and MER for both. Optional: needs an OpenAI API key
  (`.env.example`) and `requirements-gpt.txt`.
- `analyses/archetype_table.py` (run by `make sup-clustering`): S1 Table 6 with the number of
  meals, number of foods, top food categories and largest absolute Cohen's d per archetype.

## v1.0.0 (2026-11-13)

First release, accompanying Chan & Tagkopoulos (2026), *PLOS Digital Health* 5(5): e0001367
([doi:10.1371/journal.pdig.0001367](https://doi.org/10.1371/journal.pdig.0001367)).

- `make reproduce` runs the full pipeline from the raw NHANES recall files: preprocessing, the 34
  meal archetypes (Figure 2), the CVAE with 5-fold cross-validation (S1 Table 7), RDI evaluation
  (Figure 3), substitutions and the θ sweep (Figure 4), and the headline-numbers log.
- `make supplementary` and `make condition-ablation` run the supplementary analyses
  (S1 Tables 11–19).
- Every stage is deterministic in a fixed environment; package versions are pinned in
  `requirements.txt`.
- Released under the MIT License.
