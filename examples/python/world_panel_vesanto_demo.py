"""
World Panel Vesanto Demo: 3-Year Sliding Windows + Per-Decade Clustering

Workflow:
1. Load World Bank economic indicators (176 countries, 1990-2023)
2. Create 3-year sliding windows (step=2, no normalization)
3. Train SE-DBGSOM with Bayesian optimization on ALL windows
4. Per-decade Vesanto clustering: for each decade (1990, 2000, 2010, 2020),
   gather ALL active neurons from every even year in that decade, cluster once
5. For each even year snapshot: assign countries their BMU's decade cluster
6. Export all raw data as CSV for site visualization

Snapshot years (every even year): 1990, 1992, ..., 2020
Decade clustering boundaries: 1990, 2000, 2010, 2020

Run from project root:
    python examples/python/world_panel_vesanto_demo.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import warnings
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
WINDOW_SIZE = 3
STEP_SIZE = 2

SNAPSHOT_YEARS = list(range(1990, 2022, 2))  # every even year: 1990, 1992, ..., 2020
DECADE_YEARS = [1990, 2000, 2010, 2020]

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
    """Map a snapshot year to its clustering decade."""
    for d in reversed(DECADE_YEARS):
        if year >= d:
            return d
    return DECADE_YEARS[0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_and_prepare_data(
    data_path: Path,
    year_start: int = 1990,
    year_end: int = 2023
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and prepare the world panel data."""
    df = pd.read_csv(data_path)

    cols_to_keep = ['Country Code', 'Country Name', 'Year'] + FEATURE_COLS
    df = df[[c for c in cols_to_keep if c in df.columns]]

    df = df[(df['Year'] >= year_start) & (df['Year'] <= year_end)]

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    print(f"Loaded {len(df)} observations from {df['Country Code'].nunique()} countries")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Features: {len(feature_cols)}")

    return df, feature_cols


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------
def create_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
    """
    Create sliding windows from panel data (no normalization).

    Returns (X, entity_ids, time_indices, window_feature_names) where
    time_indices contains the END year of each window.
    """
    print(f"\nCreating sliding windows (window_size={window_size}, step={step_size}, no normalization)...")

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

    X = np.array(windows)

    window_feature_names = []
    for t_offset in range(window_size):
        t_label = t_offset - window_size + 1
        suffix = f"_t{t_label}" if t_label != 0 else "_t0"
        for fname in feature_cols:
            window_feature_names.append(fname + suffix)

    print(f"Created {X.shape[0]} windows with {X.shape[1]} features each")
    print(f"Unique entities: {len(set(entity_ids))}")

    return X, entity_ids, time_indices, window_feature_names


# ---------------------------------------------------------------------------
# Vesanto per-decade clustering (active neurons only)
# ---------------------------------------------------------------------------
def _cluster_active_neurons(
    active_positions: List[Tuple],
    all_positions: List[Tuple],
    all_weights: np.ndarray,
    pos_to_idx: Dict,
    min_clusters: int = 2,
    max_clusters: int = 15,
) -> Tuple[Dict, int, Dict, Dict]:
    """
    Vesanto-style Ward clustering on ACTIVE neurons only.

    Clusters only the neurons that are BMUs for a given snapshot,
    so each decade gets its own independent cluster structure.

    Returns (neuron_labels, optimal_k, db_scores, sil_scores).
    """
    active_list = sorted(set(active_positions), key=lambda p: (p[0], p[1]))
    n_neurons = len(active_list)

    if n_neurons < 2:
        labels = {pos: 1 for pos in active_list}
        return labels, 1, {}, {}

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
        optimal_k = min_clusters
    else:
        optimal_k = min(db_scores.keys(), key=db_scores.get)

    cluster_ids = fcluster(linkage_matrix, optimal_k, criterion='maxclust')
    labels = {pos: int(cluster_ids[i]) for i, pos in enumerate(active_list)}

    return labels, optimal_k, db_scores, sil_scores


def create_vesanto_snapshots(
    som,
    X: np.ndarray,
    entity_ids: List[str],
    time_indices: List[int],
    df: pd.DataFrame,
    window_feature_names: List[str],
    output_dir: Path
) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """
    Two-stage per-decade Vesanto clustering + snapshot assignment.

    Stage A: For each decade, gather active neurons from ALL even years
             in that decade. Cluster those neurons once (Ward, auto-k).
    Stage B: For each even-year snapshot, assign countries their BMU's
             cluster from the appropriate decade.

    Returns (traj_df, evaluation_summary, decade_results).
    """
    print("\nPer-decade Vesanto clustering (two-stage)...")

    bmus = som.predict(X)
    positions, weights = som.get_neuron_weights()
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

    # ---- Build full dataframe of all windows ----
    country_names = (
        df.drop_duplicates('Country Code')
        .set_index('Country Code')['Country Name']
        .to_dict()
    )

    all_rows = []
    for i, (entity, year, bmu) in enumerate(zip(entity_ids, time_indices, bmus)):
        row = {
            'country_code': entity,
            'country_name': country_names.get(entity, entity),
            'time_index': year,
            'bmu_x': bmu[0],
            'bmu_y': bmu[1],
            'window_idx': i,
        }
        for j, fname in enumerate(window_feature_names):
            row[fname] = X[i, j]
        all_rows.append(row)

    full_df = pd.DataFrame(all_rows)

    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = output_dir / "vesanto_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Stage A: Cluster each decade independently
    # ==================================================================
    print("\nStage A: Clustering each decade independently...")
    decade_results = {}  # decade -> {labels, k, db_scores, sil_scores}

    for decade in DECADE_YEARS:
        next_decade = decade + 10
        decade_snapshot_years = [
            y for y in SNAPSHOT_YEARS if decade <= y < next_decade
        ]

        # Gather BMUs from ALL even years in this decade
        all_decade_bmus = []
        for sy in decade_snapshot_years:
            end_year = sy + WINDOW_SIZE - 1
            mask = full_df['time_index'] == end_year
            idxs = full_df[mask]['window_idx'].values
            all_decade_bmus.extend([bmus[i] for i in idxs])

        # Cluster active neurons for this decade
        labels, k, db_scores, sil_scores = _cluster_active_neurons(
            all_decade_bmus, positions, weights, pos_to_idx
        )

        decade_results[decade] = {
            'labels': labels,
            'k': k,
            'db_scores': db_scores,
            'sil_scores': sil_scores,
        }

        # Save k-evaluation for this decade
        k_eval_rows = []
        all_ks = sorted(set(list(db_scores.keys()) + list(sil_scores.keys())))
        for kval in all_ks:
            k_eval_rows.append({
                'k': kval,
                'davies_bouldin_score': db_scores.get(kval, np.nan),
                'silhouette_score': sil_scores.get(kval, np.nan),
            })
        if k_eval_rows:
            pd.DataFrame(k_eval_rows).to_csv(
                eval_dir / f"k_evaluation_{decade}.csv", index=False
            )

        snap_db = db_scores.get(k, np.nan)
        snap_sil = sil_scores.get(k, np.nan)
        db_str = f"{snap_db:.3f}" if not np.isnan(snap_db) else "N/A"
        sil_str = f"{snap_sil:.3f}" if not np.isnan(snap_sil) else "N/A"
        n_active = len(set(all_decade_bmus))
        print(f"  {decade}s: {n_active} active neurons, "
              f"k={k}, DB={db_str}, Sil={sil_str}")

    # ==================================================================
    # Stage B: Assign snapshots for ALL even years
    # ==================================================================
    print("\nStage B: Assigning decade clusters to all snapshots...")
    evaluation_summary = []
    snapshot_frames = []

    for snapshot_year in SNAPSHOT_YEARS:
        decade = get_decade(snapshot_year)
        labels_map = decade_results[decade]['labels']
        decade_k = decade_results[decade]['k']
        decade_db = decade_results[decade]['db_scores']
        decade_sil = decade_results[decade]['sil_scores']

        end_year = snapshot_year + WINDOW_SIZE - 1
        year_mask = full_df['time_index'] == end_year
        year_data = full_df[year_mask].copy()

        n_countries = len(year_data)
        year_data['snapshot_year'] = snapshot_year
        year_data['decade'] = decade

        if n_countries < 2:
            year_data['cluster'] = 1
            evaluation_summary.append({
                'snapshot_year': snapshot_year,
                'time_index': end_year,
                'decade': decade,
                'decade_k': 1,
                'n_clusters': 0,
                'n_countries': n_countries,
                'n_active_neurons': 0,
                'db_score': np.nan,
                'sil_score': np.nan,
            })
            snapshot_frames.append(year_data)
            continue

        # Assign decade cluster labels via BMU lookup
        year_indices = year_data['window_idx'].values
        year_bmus = [bmus[i] for i in year_indices]
        cluster_list = [labels_map.get(bmu, 1) for bmu in year_bmus]
        year_data['cluster'] = cluster_list

        active_set = sorted(set(year_bmus), key=lambda p: (p[0], p[1]))
        n_active = len(active_set)
        n_clusters = len(set(cluster_list))

        snap_db = decade_db.get(decade_k, np.nan)
        snap_sil = decade_sil.get(decade_k, np.nan)

        print(f"  {snapshot_year} (decade {decade}s): "
              f"{n_countries} countries, {n_active} active neurons, "
              f"k={decade_k}")

        evaluation_summary.append({
            'snapshot_year': snapshot_year,
            'time_index': end_year,
            'decade': decade,
            'decade_k': decade_k,
            'n_clusters': n_clusters,
            'n_countries': n_countries,
            'n_active_neurons': n_active,
            'db_score': snap_db,
            'sil_score': snap_sil,
        })

        # Save snapshot CSV
        snapshot_cols = (
            ['country_code', 'country_name', 'snapshot_year', 'decade',
             'bmu_x', 'bmu_y', 'cluster']
            + window_feature_names
        )
        year_data[snapshot_cols].to_csv(
            snapshots_dir / f"snapshot_{snapshot_year}.csv", index=False
        )

        snapshot_frames.append(year_data)

    # Combine all snapshots into trajectory dataframe
    traj_df = pd.concat(snapshot_frames, ignore_index=True)
    traj_cols = (
        ['country_code', 'country_name', 'snapshot_year', 'time_index',
         'decade', 'bmu_x', 'bmu_y', 'cluster']
        + window_feature_names
    )
    traj_df = traj_df[traj_cols]

    print(f"\nSaved {len(SNAPSHOT_YEARS)} snapshot CSVs to {snapshots_dir}")

    return traj_df, evaluation_summary, decade_results


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------
def save_vesanto_evaluation(
    evaluation_summary: List[Dict],
    output_dir: Path
):
    """Save the cluster evaluation summary CSV."""
    eval_dir = output_dir / "vesanto_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    eval_df = pd.DataFrame(evaluation_summary)
    path = eval_dir / "cluster_evaluation_summary.csv"
    eval_df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_all_trajectories(
    traj_df: pd.DataFrame,
    output_dir: Path
):
    """Save trajectory data: combined, per-country, and summary."""
    print("\nSaving all country trajectories...")

    raw_data_dir = output_dir / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Full combined trajectories
    traj_df.to_csv(raw_data_dir / "all_trajectories.csv", index=False)
    print(f"Saved: {raw_data_dir / 'all_trajectories.csv'}")

    # Per-country trajectory files
    trajectories_dir = output_dir / "country_trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    countries = traj_df['country_code'].unique()
    for country in countries:
        country_data = traj_df[traj_df['country_code'] == country].sort_values('snapshot_year')
        safe_name = country.replace('/', '_').replace('\\', '_')
        country_data.to_csv(trajectories_dir / f"trajectory_{safe_name}.csv", index=False)

    print(f"Saved {len(countries)} individual country trajectories to {trajectories_dir}")

    # Country summary
    summary_rows = []
    for country in countries:
        cdata = traj_df[traj_df['country_code'] == country].sort_values('snapshot_year')
        clusters = cdata['cluster'].values
        # Count cluster changes only within same decade (cross-decade changes are expected)
        n_changes = 0
        if len(cdata) > 1:
            decades = cdata['decade'].values
            for idx in range(1, len(clusters)):
                if decades[idx] == decades[idx - 1] and clusters[idx] != clusters[idx - 1]:
                    n_changes += 1

        summary_rows.append({
            'country_code': country,
            'country_name': cdata['country_name'].iloc[0],
            'first_snapshot': int(cdata['snapshot_year'].min()),
            'last_snapshot': int(cdata['snapshot_year'].max()),
            'n_snapshots': len(cdata),
            'start_bmu_x': cdata['bmu_x'].iloc[0],
            'start_bmu_y': cdata['bmu_y'].iloc[0],
            'end_bmu_x': cdata['bmu_x'].iloc[-1],
            'end_bmu_y': cdata['bmu_y'].iloc[-1],
            'dominant_cluster': int(cdata['cluster'].mode().iloc[0]),
            'n_cluster_changes': n_changes,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(raw_data_dir / "country_summary.csv", index=False)
    print(f"Saved: {raw_data_dir / 'country_summary.csv'}")


def save_neuron_weights(
    som,
    X: np.ndarray,
    window_feature_names: List[str],
    output_dir: Path
):
    """Save neuron positions, weight vectors, and hit counts."""
    raw_data_dir = output_dir / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    positions, weights = som.get_neuron_weights()
    hit_counts = som.get_hit_counts(X)

    neuron_rows = []
    for i, pos in enumerate(positions):
        row = {'neuron_x': pos[0], 'neuron_y': pos[1]}
        for j, fname in enumerate(window_feature_names[:weights.shape[1]]):
            row[fname] = weights[i, j]
        row['hit_count'] = hit_counts.get(pos, 0)
        neuron_rows.append(row)

    neuron_df = pd.DataFrame(neuron_rows)
    path = raw_data_dir / "neuron_weights.csv"
    neuron_df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_neuron_clusters(
    decade_results: Dict,
    output_dir: Path
):
    """Save per-decade neuron-to-cluster mapping for site boundary drawing."""
    raw_data_dir = output_dir / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for decade in sorted(decade_results.keys()):
        labels = decade_results[decade]['labels']
        for pos, cluster_id in sorted(labels.items(), key=lambda x: (x[0][0], x[0][1])):
            rows.append({
                'decade': decade,
                'neuron_x': pos[0],
                'neuron_y': pos[1],
                'cluster_id': cluster_id,
            })

    nc_df = pd.DataFrame(rows)
    path = raw_data_dir / "neuron_clusters.csv"
    nc_df.to_csv(path, index=False)
    print(f"Saved: {path}  ({len(rows)} neuron-decade assignments)")


def save_cluster_names(
    decade_results: Dict,
    all_weights: np.ndarray,
    pos_to_idx: Dict,
    window_feature_names: List[str],
    output_dir: Path
):
    """Auto-generate per-decade cluster names from GDP centroids."""
    raw_data_dir = output_dir / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Find the GDP t0 feature index
    gdp_idx = None
    for i, fname in enumerate(window_feature_names):
        if 'GDP_per_capita' in fname and fname.endswith('_t0'):
            gdp_idx = i
            break

    tier_labels = [
        'Least Developed',
        'Low Income',
        'Lower-Middle Income',
        'Middle Income',
        'Upper-Middle Income',
        'High Income',
        'Advanced Economies',
    ]

    rows = []
    for decade in sorted(decade_results.keys()):
        labels = decade_results[decade]['labels']
        cluster_ids = sorted(set(labels.values()))

        # Compute GDP centroid for each cluster
        cluster_gdp = {}
        for cid in cluster_ids:
            cluster_positions = [p for p, l in labels.items() if l == cid]
            cluster_weights = np.array(
                [all_weights[pos_to_idx[p]] for p in cluster_positions]
            )
            centroid = cluster_weights.mean(axis=0)
            cluster_gdp[cid] = centroid[gdp_idx] if gdp_idx is not None else 0.0

        # Sort clusters by GDP centroid and assign tiered names
        sorted_clusters = sorted(cluster_gdp.items(), key=lambda x: x[1])
        n = len(sorted_clusters)

        for rank, (cid, gdp_val) in enumerate(sorted_clusters):
            tier_idx = min(int(rank * len(tier_labels) / n), len(tier_labels) - 1)
            name = tier_labels[tier_idx]

            # Deduplicate names within a decade by appending rank
            count = sum(1 for r in rows if r['decade'] == decade and r['cluster_name'] == name)
            if count > 0:
                name = f"{name} ({count + 1})"

            rows.append({
                'decade': decade,
                'cluster_id': cid,
                'cluster_name': name,
                'gdp_centroid': round(gdp_val, 2),
                'n_neurons': sum(1 for l in labels.values() if l == cid),
            })

    cn_df = pd.DataFrame(rows)
    path = raw_data_dir / "cluster_names.csv"
    cn_df.to_csv(path, index=False)
    print(f"Saved: {path}  ({len(rows)} cluster-decade names)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("   World Panel Vesanto Demo: 3-Year Windows + Per-Decade Clustering")
    print("=" * 70)
    print()

    # Paths
    data_path = Path(__file__).parent.parent.parent / "world_panel" / "world_panel_cleaned.csv"
    output_dir = Path(__file__).parent.parent / "output" / "world_panel_vesanto"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("-" * 70)
    print("Step 1: Loading World Panel Data")
    print("-" * 70)

    df, feature_cols = load_and_prepare_data(data_path)

    # =========================================================================
    # Step 2: Create Sliding Windows (3 years, step 2, no normalization)
    # =========================================================================
    print("-" * 70)
    print("Step 2: Creating Sliding Windows (3 years, step 2, raw data)")
    print("-" * 70)

    X, entity_ids, time_indices, window_feature_names = create_sliding_windows(
        df, feature_cols, window_size=WINDOW_SIZE, step_size=STEP_SIZE
    )
    print(f"\nFinal data shape: {X.shape}")
    print(f"  NaN values: {np.isnan(X).sum()} ({np.isnan(X).mean():.1%} of entries)")

    # =========================================================================
    # Step 3: Train SE-DBGSOM with Bayesian Optimization
    # =========================================================================
    print("-" * 70)
    print("Step 3: Training SE-DBGSOM with Bayesian Optimization")
    print("-" * 70)

    som = SEDBGSOM(
        bayesian=True,
        bayesian_trials=40,
        bayesian_te_constraint=0.30,
        bayesian_ranges={
            'lambda_': (1, 15),
            'max_neurons': (80, 120),
            'n_iter': (200, 300),
            'init_size': (2, 5),
        },
        random_state=27,
    )
    som.fit(X)

    opt = som.optimization_result_
    print()
    print("Training Results:")
    print(f"  Neurons: {som.n_neurons_}")
    print(f"  Lambda:  {som.lambda_:.3f}")
    print(f"  QE:      {opt['best_qe']:.4f}")
    print(f"  TE:      {opt['best_te']:.4f}")

    # =========================================================================
    # Step 4: Per-Decade Vesanto Clustering + Snapshot Assignment
    # =========================================================================
    print("-" * 70)
    print("Step 4: Per-Decade Vesanto Clustering (4 decades)")
    print("-" * 70)

    traj_df, eval_summary, decade_results = create_vesanto_snapshots(
        som, X, entity_ids, time_indices, df, window_feature_names, output_dir
    )

    # =========================================================================
    # Step 5: Save All Data
    # =========================================================================
    print("-" * 70)
    print("Step 5: Saving All Data")
    print("-" * 70)

    positions, weights = som.get_neuron_weights()
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

    save_vesanto_evaluation(eval_summary, output_dir)
    save_all_trajectories(traj_df, output_dir)
    save_neuron_weights(som, X, window_feature_names, output_dir)
    save_neuron_clusters(decade_results, output_dir)
    save_cluster_names(decade_results, weights, pos_to_idx,
                       window_feature_names, output_dir)

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  SOM neurons:       {som.n_neurons_}")
    print(f"  Training samples:  {X.shape[0]}")
    print(f"  Feature dimension: {X.shape[1]}")
    print(f"  Snapshots:         {len(SNAPSHOT_YEARS)}")
    print(f"  Decades:           {DECADE_YEARS}")
    print()
    print("  Per-Decade Clustering:")
    print(f"  {'Decade':<10} {'k':>5} {'Neurons':>10} {'DB':>8} {'Sil':>8}")
    print("  " + "-" * 43)
    for decade in DECADE_YEARS:
        dr = decade_results[decade]
        db_val = dr['db_scores'].get(dr['k'], np.nan)
        sil_val = dr['sil_scores'].get(dr['k'], np.nan)
        db_str = f"{db_val:.3f}" if not np.isnan(db_val) else "N/A"
        sil_str = f"{sil_val:.3f}" if not np.isnan(sil_val) else "N/A"
        n_active = len(dr['labels'])
        print(f"  {decade}s     {dr['k']:>5} {n_active:>10} {db_str:>8} {sil_str:>8}")
    print()
    print("  Per-Snapshot Country Counts:")
    print(f"  {'Year':<8} {'Decade':<10} {'Countries':>10}")
    print("  " + "-" * 30)
    for entry in eval_summary:
        print(f"  {entry['snapshot_year']:<8} {entry['decade']:<10} "
              f"{entry['n_countries']:>10}")
    print()
    print(f"  Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
