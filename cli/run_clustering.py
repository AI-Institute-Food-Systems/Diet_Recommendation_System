#!/usr/bin/env python3
"""
Cluster the NHANES meals of each meal type and draw the cluster figures (`make clustering`).

Reads the preprocessed meal matrices (data/processed/real_meals/dataframes/<meal>_bci_filtered.pkl,
via utils.helpers.load_meal_data) and data/mappings/food_to_nutrient.pkl. The Makefile runs,
from the repository root:

    python3 cli/run_clustering.py --out-dir outputs/clustering --feature-type hybrid \
        --normalize zscore --method enhanced_hdbscan --clear-cache \
        --viz-out outputs/clustering --gray-fraction 0.2

The t-SNE embeddings of the figures use perplexity 64 and seed 42, fixed in utils/clustering_viz.py.

Engine (utils.clustering_engine.MealClusterEngine): meal features, HDBSCAN per meal type,
cluster statistics and reports, and the per-cluster gram matrices that train the CVAE
(<out-dir>/<meal>/cluster_<id>_training_data.pkl). Figures (utils.clustering_viz.MealClusterViz):
per meal type <meal>_cluster_landscape and <meal>_dot_matrix, and Figure 2,
combined_tsne_landscape.png/.svg. stdout and stderr go to logs/clustering/clustering.log.
"""
import argparse, os, sys
import logging
from pathlib import Path

sys.path.append('src')
from utils.helpers import setup_logger as _setup_logger

# File-like object that sends print() output to the clustering log (see main()).
class _StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int):
        self._logger = logger
        self._level = level

    def write(self, buf: str):
        for line in buf.rstrip().splitlines():
            self._logger.log(self._level, line.rstrip())

    def flush(self):
        pass

from utils.clustering_engine import MealClusterEngine
from utils.clustering_viz import MealClusterViz
from utils.helpers import load_meal_data, load_nutrient_data

# ---------------------------- main ------------------------------
def main():
    ap = argparse.ArgumentParser(description='Run meal clustering (engine) + visualization')
    # engine
    ap.add_argument('--clear-cache', action='store_true',
                    help='Remove cached feature PKLs under --out-dir (not clustering result cache)')
    ap.add_argument('--cache-info', action='store_true')
    ap.add_argument('--out-dir', default='outputs/clustering',
                    help='Directory for engine outputs (features, cVAE exports, clustering cache, _tsne_cache, reports)')
    ap.add_argument('--no-caches', action='store_true',
                    help='Recompute features and t-SNE in memory only (no feature/tsne/result pickle reads or writes)')
    ap.add_argument('--feature-type', choices=['nutritional','hybrid'], default='hybrid')
    ap.add_argument('--normalize', choices=['zscore','minmax'], default='zscore')
    ap.add_argument('--method', choices=['enhanced_hdbscan','hdbscan','hierarchical','kmeans'],
                    default='enhanced_hdbscan')
    ap.add_argument('--min-cluster-size', type=int, default=200)
    ap.add_argument('--min-samples', type=int, default=100)
    ap.add_argument('--no-composition', action='store_true',
                        help='Exclude composition features (diversity, balance, counts, densities, group ratios)')
    ap.add_argument('--no-transforms', action='store_true',
                        help='Exclude transformed features (log_* of nutrients)')
    ap.add_argument('--load-cache', action='store_true',
                        help='Load clustering results from cache if present (default: recompute)')
    # viz
    ap.add_argument('--no-visuals', action='store_true', help='Skip the visualization phase')
    ap.add_argument('--viz-out', default=None, help='Output dir for figures (default: same as --out-dir)')
    ap.add_argument('--noise-fraction', type=float, default=1.0,
                        help='Fraction of noise points (-1 label) to render as gray in t-SNE (e.g., 0.1 for 10%%)')
    ap.add_argument('--gray-fraction', type=float, default=1.0,
                        help='Fraction of all gray background points to render (includes non-allowed clusters)')

    args = ap.parse_args()

    # From here on, stdout/stderr go to logs/clustering/clustering.log.
    try:
        log_path = Path.cwd() / 'logs' / 'clustering' / 'clustering.log'
        os.makedirs(log_path.parent, exist_ok=True)
        logger = _setup_logger('clustering', str(log_path))
        sys.stdout = _StreamToLogger(logger, logging.INFO)
        sys.stderr = _StreamToLogger(logger, logging.ERROR)
    except Exception:
        pass

    args.out_dir = os.path.normpath(args.out_dir)
    if args.viz_out is None:
        args.viz_out = args.out_dir
    else:
        args.viz_out = os.path.normpath(args.viz_out)

    print("Starting meal clustering analysis (engine/viz split)...")

    # ---- data ----
    meal_data = load_meal_data()
    for meal_type, meals in meal_data.items():
        print(f"Loaded {len(meals)} {meal_type} meals")
    nutrient_data = load_nutrient_data()
    print(f"Loaded nutrient data for {len(nutrient_data)} foods")

    # Assert all meal columns are mappable via nutrient PKL
    try:
        allowed = set(nutrient_data.index.astype(str).str.zfill(8))
        for meal_type, df in meal_data.items():
            cols = [str(c).zfill(8) for c in df.columns]
            missing = sorted(set(cols) - allowed)
            assert not missing, (
                f"Unmappable food codes detected in {meal_type} dataset: {missing[:20]} (total {len(missing)})\n"
                f"Ensure preprocessing meal_matrix outputs are aligned with nutrient PKL."
            )
        print("✓ All meal feature codes are mappable to nutrient table (PKL)")
    except Exception as e:
        print(f"[ASSERT] Mappability check failed: {e}")
        raise

    # ---- engine ----
    engine = MealClusterEngine(meal_data, nutrient_data, output_dir=args.out_dir)

    # Configure feature toggles
    engine.include_composition = not args.no_composition
    engine.include_transforms = not args.no_transforms

    if args.cache_info:
        engine.get_cache_info()
        return

    if args.clear_cache:
        print("Clearing all cached features..."); engine.clear_feature_cache(); print("Cache cleared.")

    cache_file = os.path.join(args.out_dir, "cached_clustering.pkl")
    loaded_cache = False
    if not args.no_caches and args.load_cache and os.path.exists(cache_file) and not args.clear_cache:
        print("Found cached clustering results. Loading...")
        loaded_cache = engine.load_cached_results(cache_file)
    elif not args.no_caches and os.path.exists(cache_file):
        print("Skipping cached clustering results (default). For cached load, pass --load-cache.")

    use_feature_cache = not args.no_caches

    if not loaded_cache:
        print("\n=== ENGINE PIPELINE ===")
        engine.prepare_meal_features(feature_type=args.feature_type,
                                     normalize_method=args.normalize,
                                     use_cache=use_feature_cache)
        engine.perform_clustering_by_meal_type(method=args.method,
                                               min_cluster_size=args.min_cluster_size,
                                               min_samples=args.min_samples)
        engine.validate_clusters_statistically()
        engine.characterize_clusters()
        engine.generate_feature_contribution_report()
        engine.diagnose_clustering_issues()

        # export grams for cVAE
        for meal_type in engine.meal_types:
            if meal_type in engine.meal_type_clusters:
                labels = engine.meal_type_clusters[meal_type]['labels']
                engine.export_cvae_training_matrices(meal_type, labels)

        if not args.no_caches:
            engine.save_cached_results(cache_file)
            print(f"Engine results saved to: {cache_file}")
        else:
            print("Skipping on-disk clustering result cache (--no-caches).")

    # ---- viz ----
    if not args.no_visuals:
        print("\n=== VISUALIZATION PHASE ===")
        viz = MealClusterViz(engine, use_tsne_cache=not args.no_caches)
        # Apply noise subsampling for plots (only affects background noise points)
        try:
            viz.filter.noise_fraction = max(0.0, min(1.0, float(args.noise_fraction)))
        except Exception:
            viz.filter.noise_fraction = 1.0
        # Apply overall gray background subsampling (after noise sampling)
        try:
            viz.filter.gray_fraction = max(0.0, min(1.0, float(args.gray_fraction)))
        except Exception:
            viz.filter.gray_fraction = 1.0

        for meal_type in engine.meal_types:
            if meal_type not in engine.meal_type_clusters:
                continue

            # t-SNE landscape with category donuts, plus the separate dot-matrix figure
            print(f"  • {meal_type}: composing aggregate figure...")
            viz.compose_cluster_landscape(
                meal_type,
                outdir=args.viz_out,
                dpi=300,
            )
        
        # Create combined t-SNE plot
        print("  • Creating combined t-SNE plot...")
        viz.create_combined_tsne_plot(outdir=args.viz_out, dpi=300)
        
        print(f"Figures saved to: {args.viz_out}")

    print("\nClustering analysis complete!")
    print("Tips:")
    print("  • --cache-info to inspect feature caches under --out-dir")
    print("  • --clear-cache to remove feature PKLs under --out-dir")
    print("  • --out-dir to write artifacts without touching other clustering outputs")
    print("  • --no-caches to force full recompute with no pickle cache read/write")
    print("  • --no-visuals to skip plots if running headless")

if __name__ == "__main__":
    main()
