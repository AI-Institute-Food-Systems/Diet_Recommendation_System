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
    ] + [["-m", f"analyses.{name}", "--help"] for name in (
        "cluster_stability", "leakage_safe_clustering_cv", "lof_outlier_sensitivity",
        "meal_split_sensitivity", "portion_ablation", "nutrient_level_summary",
        "primary_outcome_cis", "cost_sensitivity", "substitution_outcomes", "condition_ablation",
    )]

    def test_help(self) -> None:
        for args in self.ENTRY_POINTS:
            with self.subTest(entry_point=" ".join(args)):
                r = _run(args)
                self.assertEqual(r.returncode, 0, msg=(r.stderr or r.stdout)[-3000:])


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
