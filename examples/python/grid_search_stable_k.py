"""
Grid Search for Stable K across Decades

Searches over SEDBGSOM parameters to find configurations that produce
the most stable number of clusters (k) across decades in the World Panel demo.

Stability metrics:
- k_range: max(k) - min(k) across decades (lower is better)
- k_std: standard deviation of k across decades (lower is better)
- k_cv: coefficient of variation = std/mean (lower is better)

Run from project root:
    python examples/python/grid_search_stable_k.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import warnings
import itertools
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from dbgsom import SEDBGSOM
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import davies_bouldin_score, silhouette_score

# ---------------------------------------------------------------------------
# Constants (same as world_panel_vesanto_demo.py)
# ---------------------------------------------------------------------------
WINDOW_SIZE = 3
STEP_SIZE = 2

SNAPSHOT_YEARS = list(range(1980, 2024, 2))
DECADE_YEARS = [1980, 1990, 2000, 2010, 2020]

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


def get_decade(year: int) -> int:
    for d in reversed(DECADE_YEARS):
        if year >= d:
            return d
    return DECADE_YEARS[0]


def load_and_prepare_data(data_path: Path, year_start: int = 1980) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(data_path)
    cols_to_keep = ['Country Code', 'Country Name', 'Year'] + FEATURE_COLS
    df = df[[c for c in cols_to_keep if c in df.columns]]
    df = df[(df['Year'] >= year_start) & (df['Year'] <= 2023)]
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    return df, feature_cols


def create_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, List[str], List[int]]:
    windows = []
    entity_ids = []
    time_indices = []

    for country_code, country_df in df.groupby('Country Code'):
        country_df = country_df.sort_values('Year').reset_index(drop=True)
        n_years = len(country_df)
        if n_years < WINDOW_SIZE:
            continue
        features = country_df[feature_cols].values
        years = country_df['Year'].values
        for t in range(WINDOW_SIZE - 1, n_years, STEP_SIZE):
            window_start = t - WINDOW_SIZE + 1
            window_data = features[window_start:t + 1]
            window_vec = window_data.flatten()
            windows.append(window_vec)
            entity_ids.append(country_code)
            time_indices.append(int(years[t]))

    return np.array(windows), entity_ids, time_indices


def cluster_active_neurons(
    active_positions: List[Tuple],
    all_weights: np.ndarray,
    pos_to_idx: Dict,
    min_clusters: int = 2,
    max_clusters: int = 15,
) -> Tuple[int, float, float]:
    """Cluster active neurons and return (optimal_k, db_score, sil_score)."""
    active_list = sorted(set(active_positions), key=lambda p: (p[0], p[1]))
    n_neurons = len(active_list)

    if n_neurons < 2:
        return 1, np.nan, np.nan

    active_weights = np.array([all_weights[pos_to_idx[p]] for p in active_list])
    distances = pdist(active_weights, metric='euclidean')
    linkage_matrix = linkage(distances, method='ward')

    max_k = min(max_clusters, n_neurons - 1)
    db_scores = {}
    sil_scores = {}

    for k in range(min_clusters, max_k + 1):
        cluster_ids = fcluster(linkage_matrix, k, criterion='maxclust')
        if len(set(cluster_ids)) < 2:
            continue
        try:
            db_scores[k] = davies_bouldin_score(active_weights, cluster_ids)
            sil_scores[k] = silhouette_score(active_weights, cluster_ids)
        except (ValueError, ZeroDivisionError):
            continue

    if not db_scores:
        return min_clusters, np.nan, np.nan

    optimal_k = min(db_scores.keys(), key=db_scores.get)
    return optimal_k, db_scores.get(optimal_k, np.nan), sil_scores.get(optimal_k, np.nan)


def evaluate_params(
    X: np.ndarray,
    entity_ids: List[str],
    time_indices: List[int],
    lambda_: float,
    max_neurons: int,
    n_iter: int,
    init_size: Tuple[int, int],
    random_state: int,
) -> Dict:
    """Train SOM with given params and return k stability metrics."""
    try:
        som = SEDBGSOM(
            lambda_=lambda_,
            max_neurons=max_neurons,
            n_iter=n_iter,
            init_size=init_size,
            random_state=random_state,
        )
        som.fit(X)

        bmus = som.predict(X)
        positions, weights = som.get_neuron_weights()
        pos_to_idx = {pos: i for i, pos in enumerate(positions)}

        # Build mapping of window_idx -> time_index
        window_time = {i: t for i, t in enumerate(time_indices)}

        # Per-decade clustering
        decade_k = {}
        decade_db = {}
        decade_sil = {}
        decade_neurons = {}

        for decade in DECADE_YEARS:
            next_decade = decade + 10
            decade_snapshot_years = [y for y in SNAPSHOT_YEARS if decade <= y < next_decade]

            all_decade_bmus = []
            for sy in decade_snapshot_years:
                end_year = sy + WINDOW_SIZE - 1
                for i, t in enumerate(time_indices):
                    if t == end_year:
                        all_decade_bmus.append(bmus[i])

            k, db, sil = cluster_active_neurons(all_decade_bmus, weights, pos_to_idx)
            decade_k[decade] = k
            decade_db[decade] = db
            decade_sil[decade] = sil
            decade_neurons[decade] = len(set(all_decade_bmus))

        # Stability metrics
        k_values = list(decade_k.values())
        k_range = max(k_values) - min(k_values)
        k_std = np.std(k_values)
        k_mean = np.mean(k_values)
        k_cv = k_std / k_mean if k_mean > 0 else np.inf

        return {
            'lambda_': lambda_,
            'max_neurons': max_neurons,
            'n_iter': n_iter,
            'init_size': init_size,
            'random_state': random_state,
            'n_neurons': som.n_neurons_,
            'k_1980': decade_k[1980],
            'k_1990': decade_k[1990],
            'k_2000': decade_k[2000],
            'k_2010': decade_k[2010],
            'k_2020': decade_k[2020],
            'k_mean': k_mean,
            'k_range': k_range,
            'k_std': k_std,
            'k_cv': k_cv,
            'avg_db': np.nanmean(list(decade_db.values())),
            'avg_sil': np.nanmean(list(decade_sil.values())),
            'neurons_1980': decade_neurons[1980],
            'neurons_1990': decade_neurons[1990],
            'neurons_2000': decade_neurons[2000],
            'neurons_2010': decade_neurons[2010],
            'neurons_2020': decade_neurons[2020],
            'error': None,
        }
    except Exception as e:
        return {
            'lambda_': lambda_,
            'max_neurons': max_neurons,
            'n_iter': n_iter,
            'init_size': init_size,
            'random_state': random_state,
            'error': str(e),
        }


def main():
    print("=" * 70)
    print("   Grid Search for Stable K across Decades")
    print("=" * 70)
    print()

    # Load data
    data_path = Path(__file__).parent.parent.parent / "world_panel" / "world_panel_som_ready.csv"
    output_dir = Path(__file__).parent.parent / "output" / "grid_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, feature_cols = load_and_prepare_data(data_path)
    X, entity_ids, time_indices = create_sliding_windows(df, feature_cols)
    print(f"Data shape: {X.shape}")
    print()

    # Define search space (fine-tune around current params)
    param_grid = {
        'lambda_': [3.0, 3.5, 4.0, 4.5],
        'max_neurons': [55, 60, 65],
        'n_iter': [60, 80, 100],
        'init_size': [(2, 2)],
        'random_state': [42, 123, 77],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    total = len(combinations)

    print(f"Total combinations to evaluate: {total}")
    print(f"Parameter grid:")
    for k, v in param_grid.items():
        print(f"  {k}: {v}")
    print()

    results = []
    start_time = datetime.now()

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"[{i+1}/{total}] lambda={params['lambda_']}, max_neurons={params['max_neurons']}, "
                  f"n_iter={params['n_iter']}, init={params['init_size']}, seed={params['random_state']} "
                  f"(ETA: {eta/60:.1f}min)")

        result = evaluate_params(
            X, entity_ids, time_indices,
            lambda_=params['lambda_'],
            max_neurons=params['max_neurons'],
            n_iter=params['n_iter'],
            init_size=params['init_size'],
            random_state=params['random_state'],
        )
        results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Filter out errors
    valid_df = results_df[results_df['error'].isna()].copy()

    # Sort by stability (k_range first, then k_std)
    valid_df = valid_df.sort_values(['k_range', 'k_std', 'avg_db'])

    # Save full results
    results_df.to_csv(output_dir / "grid_search_full.csv", index=False)
    print(f"\nSaved full results to {output_dir / 'grid_search_full.csv'}")

    # Print top 20 most stable configurations
    print("\n" + "=" * 70)
    print("   Top 20 Most Stable Configurations (sorted by k_range, then k_std)")
    print("=" * 70)

    cols_to_show = [
        'lambda_', 'max_neurons', 'n_iter', 'init_size', 'random_state',
        'n_neurons', 'k_1980', 'k_1990', 'k_2000', 'k_2010', 'k_2020',
        'k_range', 'k_std', 'avg_db', 'avg_sil'
    ]

    top20 = valid_df.head(20)[cols_to_show]
    print(top20.to_string(index=False))

    # Save top results
    valid_df.head(50).to_csv(output_dir / "grid_search_top50.csv", index=False)
    print(f"\nSaved top 50 to {output_dir / 'grid_search_top50.csv'}")

    # Best configuration
    if len(valid_df) > 0:
        best = valid_df.iloc[0]
        print("\n" + "=" * 70)
        print("   BEST CONFIGURATION (Most Stable K)")
        print("=" * 70)
        print(f"  lambda_:      {best['lambda_']}")
        print(f"  max_neurons:  {best['max_neurons']}")
        print(f"  n_iter:       {best['n_iter']}")
        print(f"  init_size:    {best['init_size']}")
        print(f"  random_state: {best['random_state']}")
        print()
        print(f"  Final neurons: {best['n_neurons']}")
        print(f"  K values:      1980={best['k_1980']}, 1990={best['k_1990']}, "
              f"2000={best['k_2000']}, 2010={best['k_2010']}, 2020={best['k_2020']}")
        print(f"  K range:       {best['k_range']}")
        print(f"  K std:         {best['k_std']:.3f}")
        print(f"  Avg DB score:  {best['avg_db']:.3f}")
        print(f"  Avg Silhouette:{best['avg_sil']:.3f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
