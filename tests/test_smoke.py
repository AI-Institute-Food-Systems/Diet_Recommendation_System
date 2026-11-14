"""Fast checks that need no pipeline outputs (run in CI), plus a numbers-log check
that runs only when a pipeline run exists (outputs/runs/$RUN_NAME)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = os.environ.get("RUN_NAME", "eval_run")


def _run(args: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    return subprocess.run([sys.executable, *args], cwd=REPO_ROOT, capture_output=True,
                          text=True, timeout=timeout, env=env)


class EnvironmentTest(unittest.TestCase):
    def test_requirements_installed(self) -> None:
        r = _run(["scripts/check_env.py"])
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)


class ConfigTest(unittest.TestCase):
    def test_archetypes_have_names(self) -> None:
        archetypes = json.loads((REPO_ROOT / "config/cluster_filter_default.json").read_text())
        names = json.loads((REPO_ROOT / "config/cluster_names.json").read_text(encoding="utf-8"))
        # 34 archetypes: 12 breakfast, 12 lunch, 10 dinner (S1 Table 6).
        self.assertEqual({m: len(ids) for m, ids in archetypes.items()},
                         {"Breakfast": 12, "Lunch": 12, "Dinner": 10})
        for meal, ids in archetypes.items():
            for cid in ids:
                self.assertIn(str(cid), names[meal.lower()], f"no name for {meal} cluster {cid}")

    def test_figure_registry_uses_pipeline_outputs(self) -> None:
        registry = json.loads((REPO_ROOT / "config/manuscript_figures.json").read_text())
        for fig in registry["figures"]:
            for panel in fig["panels"]:
                sources = panel["source"] if isinstance(panel["source"], list) else [panel["source"]]
                for src in sources:
                    self.assertTrue(src.startswith("outputs/"), f"{fig['id']} {panel['id']}: {src}")


class EntryPointTest(unittest.TestCase):
    """Every pipeline entry point imports its dependencies and parses arguments."""

    ENTRY_POINTS = [
        ["cli/run_clustering.py", "--help"],
        ["cli/run_multi_meal_cvae.py", "--help"],
        ["cli/evaluate.py", "--help"],
        ["-m", "experiments.analyze_meal_substitutions", "--help"],
        ["-m", "experiments.plot_substitution_figures", "--help"],
        ["-m", "experiments.substitution_foldchange.plot_nutrient_fold_changes", "--help"],
        ["scripts/generate_manuscript_numbers_log.py", "--help"],
        ["scripts/export_manuscript_figures.py", "--help"],
        ["experiments/gpt_comparison/generate_meals.py", "--help"],
        ["experiments/gpt_comparison/evaluate.py", "--help"],
    ] + [["-m", f"analyses.{name}", "--help"] for name in (
        "archetype_table", "cluster_stability", "leakage_safe_clustering_cv", "lof_outlier_sensitivity",
        "meal_split_sensitivity", "portion_ablation", "nutrient_level_summary",
        "primary_outcome_cis", "cost_sensitivity", "substitution_outcomes", "condition_ablation",
    )]

    def test_help(self) -> None:
        for args in self.ENTRY_POINTS:
            with self.subTest(entry_point=" ".join(args)):
                r = _run(args)
                self.assertEqual(r.returncode, 0, msg=(r.stderr or r.stdout)[-3000:])


class ArchetypeTableTest(unittest.TestCase):
    def test_counts_and_top_categories(self) -> None:
        import numpy as np
        import pandas as pd
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from analyses.archetype_table import archetype_rows
        labels = np.array([0, 0, 1, -1])
        meals = pd.DataFrame({"11111111": [50.0, 0, 10, 0], "22222222": [0, 30.0, 0, 5],
                              "33333333": [0, 0, 0, 80.0]})
        report = pd.DataFrame({
            "meal_type": ["breakfast"] * 4, "cluster_id": [0, 0, 0, 1],
            "feature": ["grams_pizza", "protein_g", "grams_sweetened_beverages", "grams_fruits"],
            "effect_size": [5.0, -7.5, 1.2, 2.0], "direction": ["HIGH", "LOW", "HIGH", "HIGH"],
        })
        rows = archetype_rows("Breakfast", labels, meals, report, [0], {"breakfast": {"0": "Pizza"}}, 5)
        self.assertEqual(rows, [{"meal_type": "Breakfast", "cluster_id": 0, "cluster_name": "Pizza",
                                 "n_meals": 2, "n_foods": 2,
                                 "top_categories": "Pizza; Sweetened beverages",
                                 "max_abs_cohens_d": 7.5}])


class GPTComparisonTest(unittest.TestCase):
    """Parsing and evaluation of the GPT comparison, without API access."""

    def setUp(self) -> None:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

    def test_parse_keeps_catalogue_foods_only(self) -> None:
        from experiments.gpt_comparison.generate_meals import parse_meals
        content = json.dumps({"meals": [
            {"meal_id": 1, "items": [{"code": "11111111", "grams": 100},
                                     {"code": "99999999", "grams": 50},     # not in the catalogue
                                     {"code": "22222222", "grams": 2000}]},  # grams out of range
            {"meal_id": 3, "items": [{"code": "22222222", "grams": 40}]},
        ]})
        self.assertEqual(parse_meals(content, 3, {"11111111", "22222222"}),
                         [{"11111111": 100.0}, None, {"22222222": 40.0}])
        self.assertEqual(parse_meals("not json", 2, {"11111111"}), [None, None])

    def test_foods_are_looked_up_by_code(self) -> None:
        import numpy as np
        from experiments.gpt_comparison.evaluate import evaluate
        # Three foods; nutrients per 100 g: energy, protein, fat, carbohydrate.
        names = ["energy_kcal", "protein_g", "fat_g", "carb_g"]
        nutrition = {
            "feature_names": ["11111111", "22222222", "33333333"],
            "W_per100g": np.array([[400.0, 100.0, 0.0, 0.0],     # protein only
                                   [900.0, 0.0, 100.0, 0.0],     # fat only
                                   [400.0, 0.0, 0.0, 100.0]]),   # carbohydrate only
            "nutrient_names": names,
            "RDI": np.array([2000.0, 50.0, 78.0, 275.0]),
        }
        meal = {"pair_code": 0, "meal_idx": 0, "cluster_id": 0, "sample_idx": 0, "success": True}
        results = {"cvae_meals": [meal | {"foods": {"11111111": 100.0}}],
                   "gpt_meals": [meal | {"foods": {"33333333": 50.0, "22222222": 50.0}},
                                 meal | {"success": False, "foods": None}]}
        with tempfile.TemporaryDirectory() as tmp:
            out = evaluate(results, nutrition, Path(tmp))
        self.assertAlmostEqual(out["cvae"]["protein_pct"], 100.0)
        # 50 g carbohydrate (200 kcal) and 50 g fat (450 kcal).
        self.assertAlmostEqual(out["gpt"]["carb_pct"], 100.0 * 200 / 650)
        self.assertAlmostEqual(out["gpt"]["fat_pct"], 100.0 * 450 / 650)
        self.assertEqual((out["gpt"]["n_meals"], out["gpt_meals_requested"]), (1, 2))

    def test_env_file_does_not_override_environment(self) -> None:
        from experiments.gpt_comparison.common import load_env_file
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("# comment\nGPT_TEST_A=from_file\nexport GPT_TEST_B='quoted'\n")
            os.environ["GPT_TEST_A"] = "from_env"
            try:
                load_env_file(env)
                self.assertEqual(os.environ["GPT_TEST_A"], "from_env")
                self.assertEqual(os.environ["GPT_TEST_B"], "quoted")
            finally:
                os.environ.pop("GPT_TEST_A", None)
                os.environ.pop("GPT_TEST_B", None)


class NumbersLogTest(unittest.TestCase):
    def test_numbers_log_from_existing_run(self) -> None:
        run_dir = REPO_ROOT / "outputs" / "runs" / RUN_NAME
        if not (run_dir / "tables" / "cluster_counts_totals.csv").is_file():
            self.skipTest(f"no pipeline run at {run_dir} (run `make reproduce`)")
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "final_results_log.txt"
            r = _run(["scripts/generate_manuscript_numbers_log.py", "--out-dir", str(run_dir),
                      "--out-log", str(log)])
            self.assertEqual(r.returncode, 0, msg=r.stderr or r.stdout)
            text = log.read_text(encoding="utf-8")
        for section in "ABCDEF":
            self.assertIn(f"=== Section {section} |", text)
        self.assertNotIn("Error", text)


if __name__ == "__main__":
    unittest.main()
