"""
Grid Search for SEDBGSOM + Vesanto Clustering Parameters

Searches for parameters that minimize mean Davies-Bouldin score across decades.

Run from project root:
    python examples/python/grid_search_db.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from itertools import product
import sys
import warnings
import time

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from dbgsom import SEDBGSOM
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import davies_bouldin_score, silhouette_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    'GDP_per_capita_constant_2015_US',
    'Trade_percent_of_GDP',
    'Unemployment_total_percent_of_labor_force',
    'Agriculture_value_added_percent_of_GDP',
    'Industry_value_added_percent_of_GDP',
    'Urban_population_percent',
    'Inflation_consumer_prices_annual_percent',
    'Gross_fixed_capital_formation_percent_of_GDP',
    'Access_to_electricity_percent_of_population',
    'Internet_users_percent_of_population',
    'Services_value_added_percent_of_GDP',
    'Gross_domestic_savings_percent_of_GDP',
    'Fertility_rate_total',
    'FDI_net_inflows_percent_of_GDP',
]

DECADE_YEARS = [1990, 2000, 2010, 2020]

# ---------------------------------------------------------------------------
# Parameter Grid
# ---------------------------------------------------------------------------
PARAM_GRID = {
    # SOM parameters
    'lambda_': [1.0, 1.5, 2.0, 2.5, 3.0],
    'max_neurons': [60, 80, 100, 120, 150],
    'n_iter': [150, 200, 250, 300],

    # Data parameters
    'window_size': [3, 5],
    'step_size': [1, 2],
}


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------
def load_data(data_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load world panel data."""
    df = pd.read_csv(data_path)
    cols_to_keep = ['Country Code', 'Country Name', 'Year'] + FEATURE_COLS
    df = df[[c for c in cols_to_keep if c in df.columns]]
    df = df[(df['Year'] >= 1990) & (df['Year'] <= 2023)]
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    return df, feature_cols


def create_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Create sliding windows from panel data."""
    windows = []
    entity_ids = []
    time_indices = []

    for country_code, country_df in df.groupby('Country Code'):
        country_df = country_df.sort_values('Year').reset_index(drop=True)
        n_years = len(country_df)

        if n_years < window_size:
            continue

        features = country_df[feature_cols].values
        years = country_df['Year'].values

        for t in range(window_size - 1, n_years, step_size):
            window_start = t - window_size + 1
            window_data = features[window_start:t + 1]
            window_vec = window_data.flatten()

            windows.append(window_vec)
            entity_ids.append(country_code)
            time_indices.append(int(years[t]))

    return np.array(windows), entity_ids, time_indices


# ---------------------------------------------------------------------------
# Clustering & Evaluation
# ---------------------------------------------------------------------------
def get_decade(year: int, window_size: int) -> int:
    """Map end year to snapshot year to decade."""
    snapshot_year = year - window_size + 1
    for d in reversed(DECADE_YEARS):
        if snapshot_year >= d:
            return d
    return DECADE_YEARS[0]


def cluster_and_evaluate(
    som,
    X: np.ndarray,
    time_indices: List[int],
    window_size: int,
) -> Dict[int, float]:
    """
    Perform Vesanto clustering and return DB scores per decade.
    """
    bmus = som.predict(X)
    positions, weights = som.get_neuron_weights()
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

    decade_db_scores = {}

    for decade in DECADE_YEARS:
        # Gather active neurons for this decade
        active_bmus = []
        for i, end_year in enumerate(time_indices):
            sample_decade = get_decade(end_year, window_size)
            if sample_decade == decade:
                active_bmus.append(bmus[i])

        active_positions = list(set(active_bmus))
        n_active = len(active_positions)

        if n_active < 3:
            decade_db_scores[decade] = np.nan
            continue

        # Get weights for active neurons
        active_weights = np.array([weights[pos_to_idx[p]] for p in active_positions])

        # Ward clustering
        distances = pdist(active_weights, metric='euclidean')
        linkage_matrix = linkage(distances, method='ward')

        # Find optimal k via DB score
        max_k = min(15, n_active - 1)
        best_db = np.inf

        for k in range(2, max_k + 1):
            cluster_ids = fcluster(linkage_matrix, k, criterion='maxclust')
            if len(set(cluster_ids)) < 2:
                continue
            try:
                db = davies_bouldin_score(active_weights, cluster_ids)
                if db < best_db:
                    best_db = db
            except (ValueError, ZeroDivisionError):
                continue

        decade_db_scores[decade] = best_db if best_db < np.inf else np.nan

    return decade_db_scores


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------
def run_grid_search(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Run grid search over all parameter combinations."""

    # Generate all parameter combinations
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    combinations = list(product(*param_values))

    print(f"Total combinations to test: {len(combinations)}")
    print(f"Parameters: {param_names}")
    print()

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))

        start_time = time.time()

        try:
            # Create sliding windows with current params
            X, entity_ids, time_indices = create_sliding_windows(
                df, feature_cols,
                window_size=params['window_size'],
                step_size=params['step_size'],
            )

            if len(X) < 10:
                print(f"[{i+1}/{len(combinations)}] Skipping - too few samples")
                continue

            # Train SOM
            som = SEDBGSOM(
                lambda_=params['lambda_'],
                max_neurons=params['max_neurons'],
                n_iter=params['n_iter'],
                init_size=(2, 2),
                random_state=42,
            )
            som.fit(X)

            # Evaluate clustering
            decade_db_scores = cluster_and_evaluate(
                som, X, time_indices, params['window_size']
            )

            # Compute aggregated metrics
            valid_scores = [v for v in decade_db_scores.values() if not np.isnan(v)]
            mean_db = np.mean(valid_scores) if valid_scores else np.nan
            min_db = np.min(valid_scores) if valid_scores else np.nan
            max_db = np.max(valid_scores) if valid_scores else np.nan

            elapsed = time.time() - start_time

            result = {
                **params,
                'n_neurons': som.n_neurons_,
                'n_samples': len(X),
                'db_1990': decade_db_scores.get(1990, np.nan),
                'db_2000': decade_db_scores.get(2000, np.nan),
                'db_2010': decade_db_scores.get(2010, np.nan),
                'db_2020': decade_db_scores.get(2020, np.nan),
                'mean_db': mean_db,
                'min_db': min_db,
                'max_db': max_db,
                'time_seconds': round(elapsed, 1),
            }
            results.append(result)

            print(f"[{i+1}/{len(combinations)}] "
                  f"λ={params['lambda_']}, neurons={params['max_neurons']}, "
                  f"iter={params['n_iter']}, win={params['window_size']}, "
                  f"step={params['step_size']} → "
                  f"mean_db={mean_db:.3f}, neurons={som.n_neurons_} "
                  f"({elapsed:.1f}s)")

        except Exception as e:
            print(f"[{i+1}/{len(combinations)}] Error: {e}")
            continue

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("   Grid Search: Minimize Davies-Bouldin Score")
    print("=" * 70)
    print()

    # Load data
    data_path = Path(__file__).parent.parent.parent / "world_panel" / "world_panel_som_ready.csv"
    output_dir = Path(__file__).parent.parent / "output" / "grid_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, feature_cols = load_data(data_path)
    print(f"Loaded {len(df)} observations, {len(feature_cols)} features")
    print()

    # Run grid search
    print("-" * 70)
    print("Running Grid Search")
    print("-" * 70)

    results_df = run_grid_search(df, feature_cols)

    # Save results
    results_path = output_dir / "grid_search_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to: {results_path}")

    # Report best parameters
    print()
    print("=" * 70)
    print("   Best Parameters (by mean DB score)")
    print("=" * 70)

    if len(results_df) > 0:
        # Sort by mean_db (lower is better)
        results_df = results_df.sort_values('mean_db')

        print("\nTop 10 configurations:")
        print("-" * 70)

        top_cols = ['lambda_', 'max_neurons', 'n_iter', 'window_size',
                    'step_size', 'n_neurons', 'mean_db', 'min_db', 'max_db']
        print(results_df[top_cols].head(10).to_string(index=False))

        print()
        print("Best configuration:")
        best = results_df.iloc[0]
        print(f"  lambda_:      {best['lambda_']}")
        print(f"  max_neurons:  {int(best['max_neurons'])}")
        print(f"  n_iter:       {int(best['n_iter'])}")
        print(f"  window_size:  {int(best['window_size'])}")
        print(f"  step_size:    {int(best['step_size'])}")
        print(f"  ---")
        print(f"  Final neurons: {int(best['n_neurons'])}")
        print(f"  Mean DB:       {best['mean_db']:.4f}")
        print(f"  DB range:      [{best['min_db']:.4f}, {best['max_db']:.4f}]")

        # Per-decade breakdown for best config
        print()
        print("Per-decade DB scores (best config):")
        for decade in DECADE_YEARS:
            col = f'db_{decade}'
            val = best[col]
            if not np.isnan(val):
                print(f"  {decade}s: {val:.4f}")
            else:
                print(f"  {decade}s: N/A")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
