# Changelog

## v1.0.0 (unreleased)

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
