#!/usr/bin/env python3
"""Evaluation steps run by `cli/evaluate.py` (`make eval`).

`EvaluationManager` bootstraps per-cluster nutrient medians of real vs generated meals
(src/analysis_rdi_eval/stats_bootstrap.py) and runs scripts/print_nutrient_profiles.py as a
sanity check. Both only print to the log; neither writes files.
"""

import json

import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path

from src.analysis_rdi_eval.stats_bootstrap import (
    bootstrap_cluster_boxplot_data, create_bootstrap_summary_table
)

def _get_cluster_name(meal_type: str, cluster_id: int) -> str:
    """Archetype name from config/cluster_names.json for the bootstrap progress messages (log only)."""
    try:
        path = Path(__file__).resolve().parents[2] / "config" / "cluster_names.json"
        names = json.loads(path.read_text())
        return names.get(str(meal_type).lower(), {}).get(str(int(cluster_id)), f"Cluster {cluster_id}")
    except Exception:
        return f"Cluster {cluster_id}"


class EvaluationManager:
    """Per-cluster bootstrap comparison and the nutrient-profile sanity check."""

    def __init__(self, processed: Dict, args):
        self.processed = processed
        self.args = args

    
    
    def run_bootstrap_evaluation(self, real_aligned: Dict[str, np.ndarray], 
                               gen_aligned: Dict[str, np.ndarray], 
                               clusters_aligned: Dict[str, np.ndarray],
                               adj_by_type: Optional[Dict[str, np.ndarray]] = None) -> Dict:
        """Bootstrap nutrient medians for every (meal type, cluster); `adj_by_type` is unused."""
        print("→ Bootstrapping comparisons...")
        bootstrap_results = {}
        mt_seq = [mt for mt in ["Breakfast", "Lunch", "Dinner"] if mt in real_aligned]

        for mt in mt_seq:
            Xr = real_aligned[mt]
            Xg = gen_aligned[mt]
            C  = clusters_aligned[mt]

            # Per-cluster, rows are already aligned and same count; just slice by mask
            valid_clusters = np.unique(C[C != -1])
            for k in valid_clusters:
                m = (C == k)
                Xr_k, Xg_k = Xr[m], Xg[m]

                # Safety assert
                assert len(Xr_k) == len(Xg_k), f"Aligned sizes differ for {mt} cluster {k}"

                res = bootstrap_cluster_boxplot_data(
                    Xr_k, Xg_k,
                    self.processed["W"],
                    self.processed["nutrient_names"],
                    self.processed["limits"],
                    self.processed["kcal_spec"],
                    n_boot=200,
                    sample_size=len(Xr_k),    # identical sizes by construction
                    seed=self.args.seed,
                    X_gen_adj_k=None,  # No RDI adjusted meals
                )
                if res:
                    bootstrap_results[f"{mt}_{int(k)}"] = res
                    cluster_name = _get_cluster_name(mt, int(k))
                    print(f"  ✓ Bootstrap {mt} {cluster_name}: {len(res)} nutrients")

        return bootstrap_results
    
    def create_bootstrap_summary(self, bootstrap_results: Dict) -> 'pd.DataFrame':
        """Summarize the bootstrap results (BH-adjusted p-values); printed, not saved."""
        mt_seq = [mt for mt in ["Breakfast", "Lunch", "Dinner"] if mt in self.processed["real_meals"]]
        clusters_union = sorted({int(k) for mt in mt_seq for k in np.unique(self.processed["clusters_real"].get(mt, np.array([], int))) if k != -1})
        
        # Debug: Check what keys we have vs what we're looking for
        print(f"  Bootstrap results keys: {list(bootstrap_results.keys())[:5]}...")  # Show first 5 keys
        print(f"  Looking for clusters: {clusters_union[:10]}...")  # Show first 10 clusters
        
        # Debug: Check what's actually in the first bootstrap result
        if bootstrap_results:
            first_key = list(bootstrap_results.keys())[0]
            print(f"  First bootstrap result key: {first_key}")
            print(f"  First bootstrap result type: {type(bootstrap_results[first_key])}")
            if isinstance(bootstrap_results[first_key], dict):
                print(f"  First bootstrap result keys: {list(bootstrap_results[first_key].keys())[:3]}...")
                if bootstrap_results[first_key]:
                    first_nutrient = list(bootstrap_results[first_key].keys())[0]
                    print(f"  First nutrient: {first_nutrient}")
                    print(f"  First nutrient result keys: {list(bootstrap_results[first_key][first_nutrient].keys())}")
        
        bootstrap_df = create_bootstrap_summary_table(bootstrap_results, mt_seq, clusters_union)
        
        print(f"  Bootstrap summary: {len(bootstrap_df)} rows created for {len(clusters_union)} clusters")
        return bootstrap_df
    
    def run_nutrient_profile_sanity_check(self, out_dir: Path, project_root: Path):
        """Run scripts/print_nutrient_profiles.py on `out_dir` and echo its output."""
        print("→ Running nutrient profile sanity check...")
        try:
            import subprocess
            import sys

            script = project_root / "scripts" / "print_nutrient_profiles.py"
            if not script.is_file():
                print(f"⚠ Nutrient profile script not found: {script}")
                return

            cmd = [
                sys.executable,
                str(script),
                "--out_dir",
                str(out_dir),
                "--n_samples",
                "3",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
            if result.returncode == 0:
                if result.stdout:
                    print(result.stdout)
                print("✓ Nutrient profile sanity check completed")
            else:
                detail = (result.stderr or "").strip() or (result.stdout or "").strip()
                print(f"⚠ Nutrient profile check failed: {detail}")
        except Exception as e:
            print(f"⚠ Could not run nutrient profile check: {e}")
