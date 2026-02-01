"""
World Panel Data Demo: Country Economic Trajectories with SE-DBGSOM

Demonstrates panel data workflow:
1. Load World Bank economic indicators (panel data, 1990-2022)
2. Transform to sliding windows of 2 years (no normalization)
3. Train SE-DBGSOM on windowed data
4. For each year: snapshot map state, cluster, track all country positions
5. Export full trajectories with raw data for every country

Run from project root:
    python examples/python/world_panel_demo.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from dbgsom import SEDBGSOM, cluster_leiden, assign_cluster_labels, HAS_LEIDEN

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Feature columns to use
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


def load_and_prepare_data(
    data_path: Path,
    year_start: int = 1990,
    year_end: int = 2022
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and prepare the world panel data."""
    df = pd.read_csv(data_path)

    cols_to_keep = ['Country Code', 'Country Name', 'Year'] + FEATURE_COLS
    df = df[[c for c in cols_to_keep if c in df.columns]]

    # Filter years
    df = df[(df['Year'] >= year_start) & (df['Year'] <= year_end)]

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=feature_cols, thresh=len(feature_cols) // 2)

    # Fill NaN with column median (per country, then global)
    for col in feature_cols:
        df[col] = df.groupby('Country Code')[col].transform(
            lambda x: x.fillna(x.median())
        )
    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    print(f"Loaded {len(df)} observations from {df['Country Code'].nunique()} countries")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Features: {len(feature_cols)}")

    return df, feature_cols


def create_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 2
) -> Tuple[np.ndarray, List[str], List[int], List[str]]:
    """
    Create sliding windows from panel data - NO NORMALIZATION.
    """
    print(f"\nCreating sliding windows (window_size={window_size}, no normalization)...")

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

        for t in range(window_size - 1, n_years):
            window_start = t - window_size + 1
            window_data = features[window_start:t + 1]
            window_vec = window_data.flatten()

            windows.append(window_vec)
            entity_ids.append(country_code)
            time_indices.append(int(years[t]))

    X = np.array(windows)

    # Generate feature names
    window_feature_names = []
    for t_offset in range(window_size):
        t_label = t_offset - window_size + 1
        suffix = f"_t{t_label}" if t_label != 0 else "_t0"
        for fname in feature_cols:
            window_feature_names.append(fname + suffix)

    print(f"Created {X.shape[0]} windows with {X.shape[1]} features each")
    print(f"Unique entities: {len(set(entity_ids))}")

    return X, entity_ids, time_indices, window_feature_names


def create_year_snapshots(
    som,
    X: np.ndarray,
    entity_ids: List[str],
    time_indices: List[int],
    df: pd.DataFrame,
    window_feature_names: List[str],
    output_dir: Path
) -> pd.DataFrame:
    """
    For each year, capture snapshot of all countries on the SOM.
    Cluster at each time point and save with raw feature data.
    """
    print("\nCreating yearly snapshots...")

    bmus = som.predict(X)
    country_names = df.drop_duplicates('Country Code').set_index('Country Code')['Country Name'].to_dict()

    # Build full trajectory dataframe with raw features
    all_rows = []
    for i, (entity, year, bmu) in enumerate(zip(entity_ids, time_indices, bmus)):
        row = {
            'country_code': entity,
            'country_name': country_names.get(entity, entity),
            'year': year,
            'bmu_x': bmu[0],
            'bmu_y': bmu[1],
            'window_idx': i
        }
        # Add raw feature values
        for j, fname in enumerate(window_feature_names):
            row[fname] = X[i, j]
        all_rows.append(row)

    traj_df = pd.DataFrame(all_rows)

    # Get unique years sorted
    years = sorted(traj_df['year'].unique())
    print(f"Processing {len(years)} years: {years[0]} - {years[-1]}")

    # Create snapshots directory
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # For each year, cluster and save snapshot
    year_clusters = {}

    for year in years:
        year_mask = traj_df['year'] == year
        year_data = traj_df[year_mask].copy()

        # Get indices for this year
        year_indices = year_data['window_idx'].values
        X_year = X[year_indices]

        # Cluster for this year
        if HAS_LEIDEN and len(year_indices) > 10:
            try:
                cluster_labels = cluster_leiden(som, X_year)
                # Map BMUs to clusters
                year_bmus = [bmus[i] for i in year_indices]
                year_clusters_list = [cluster_labels.get(bmu, 0) for bmu in year_bmus]
                year_data['cluster'] = year_clusters_list
            except Exception:
                year_data['cluster'] = 0
        else:
            year_data['cluster'] = 0

        year_clusters[year] = year_data

        # Save year snapshot with all raw data
        year_data.to_csv(snapshots_dir / f"snapshot_{year}.csv", index=False)

    print(f"Saved {len(years)} yearly snapshots to {snapshots_dir}")

    # Merge clusters back to main trajectory
    traj_df['cluster'] = 0
    for year, year_data in year_clusters.items():
        for _, row in year_data.iterrows():
            mask = (traj_df['country_code'] == row['country_code']) & (traj_df['year'] == row['year'])
            traj_df.loc[mask, 'cluster'] = row['cluster']

    return traj_df


def save_all_trajectories(
    traj_df: pd.DataFrame,
    output_dir: Path
):
    """Save complete trajectory data for ALL countries."""
    print("\nSaving all country trajectories...")

    # Full trajectory file
    traj_df.to_csv(output_dir / "all_trajectories.csv", index=False)
    print(f"Saved: {output_dir / 'all_trajectories.csv'}")

    # Per-country trajectory files
    trajectories_dir = output_dir / "country_trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    countries = traj_df['country_code'].unique()
    for country in countries:
        country_data = traj_df[traj_df['country_code'] == country].sort_values('year')
        safe_name = country.replace('/', '_').replace('\\', '_')
        country_data.to_csv(trajectories_dir / f"trajectory_{safe_name}.csv", index=False)

    print(f"Saved {len(countries)} individual country trajectories to {trajectories_dir}")

    # Summary per country
    summary = traj_df.groupby('country_code').agg({
        'country_name': 'first',
        'year': ['min', 'max', 'count'],
        'bmu_x': ['first', 'last'],
        'bmu_y': ['first', 'last'],
        'cluster': lambda x: x.mode().iloc[0] if len(x) > 0 else 0
    }).reset_index()
    summary.columns = [
        'country_code', 'country_name', 'year_start', 'year_end', 'n_observations',
        'start_bmu_x', 'end_bmu_x', 'start_bmu_y', 'end_bmu_y', 'dominant_cluster'
    ]
    summary.to_csv(output_dir / "country_summary.csv", index=False)
    print(f"Saved: {output_dir / 'country_summary.csv'}")


def plot_yearly_snapshots(
    som,
    traj_df: pd.DataFrame,
    output_dir: Path,
    years_to_plot: List[int] = None
):
    """Plot SOM snapshots for selected years."""
    if not HAS_MATPLOTLIB:
        return

    print("\nPlotting yearly snapshots...")

    plots_dir = output_dir / "plots" / "yearly"
    plots_dir.mkdir(parents=True, exist_ok=True)

    positions, _ = som.get_neuron_weights()

    all_years = sorted(traj_df['year'].unique())
    if years_to_plot is None:
        # Plot every 5 years + first and last
        years_to_plot = [all_years[0]] + [y for y in all_years if y % 5 == 0] + [all_years[-1]]
        years_to_plot = sorted(set(years_to_plot))

    for year in years_to_plot:
        if year not in all_years:
            continue

        year_data = traj_df[traj_df['year'] == year]

        fig, ax = plt.subplots(figsize=(12, 10))

        # Draw SOM grid
        for pos in positions:
            ax.scatter(pos[0], pos[1], c='lightgray', s=80, alpha=0.3, zorder=1)

        # Draw connections
        for pos in positions:
            for dx, dy in [(1, 0), (0, 1)]:
                neighbor = (pos[0] + dx, pos[1] + dy)
                if neighbor in [p for p in positions]:
                    ax.plot([pos[0], neighbor[0]], [pos[1], neighbor[1]],
                           'k-', alpha=0.1, linewidth=0.5, zorder=0)

        # Plot countries for this year
        n_clusters = max(1, year_data['cluster'].max())
        cmap = cm.tab20

        for _, row in year_data.iterrows():
            color = cmap(row['cluster'] / max(1, n_clusters))
            ax.scatter(row['bmu_x'], row['bmu_y'], c=[color], s=100,
                      edgecolors='black', linewidth=0.5, alpha=0.8, zorder=2)
            ax.annotate(row['country_code'], (row['bmu_x'], row['bmu_y']),
                       xytext=(3, 3), textcoords='offset points', fontsize=6, alpha=0.7)

        ax.set_xlabel('SOM Grid X')
        ax.set_ylabel('SOM Grid Y')
        ax.set_title(f'Country Positions on SOM - Year {year}\n({len(year_data)} countries)')
        plt.tight_layout()
        plt.savefig(plots_dir / f"snapshot_{year}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved yearly snapshot plots to {plots_dir}")


def plot_all_trajectories(
    som,
    traj_df: pd.DataFrame,
    output_dir: Path
):
    """Plot trajectories for ALL countries."""
    if not HAS_MATPLOTLIB:
        return

    print("\nPlotting all trajectories...")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    positions, _ = som.get_neuron_weights()

    fig, ax = plt.subplots(figsize=(16, 12))

    # Draw SOM grid
    for pos in positions:
        ax.scatter(pos[0], pos[1], c='lightgray', s=60, alpha=0.2, zorder=1)

    # Get unique countries
    countries = traj_df['country_code'].unique()
    colors = cm.tab20(np.linspace(0, 1, min(20, len(countries))))

    # Plot each country's trajectory
    for i, country in enumerate(countries):
        country_data = traj_df[traj_df['country_code'] == country].sort_values('year')
        if len(country_data) < 2:
            continue

        xs = country_data['bmu_x'].values
        ys = country_data['bmu_y'].values
        color = colors[i % 20]

        # Draw trajectory line
        ax.plot(xs, ys, '-', color=color, alpha=0.4, linewidth=1, zorder=2)

        # Mark start (circle) and end (square)
        ax.scatter(xs[0], ys[0], c=[color], s=30, marker='o', alpha=0.6, zorder=3)
        ax.scatter(xs[-1], ys[-1], c=[color], s=30, marker='s', alpha=0.6, zorder=3)

    ax.set_xlabel('SOM Grid X')
    ax.set_ylabel('SOM Grid Y')
    ax.set_title(f'All Country Trajectories ({len(countries)} countries)\nCircle=Start, Square=End')
    plt.tight_layout()
    plt.savefig(plots_dir / "all_trajectories.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plots_dir / 'all_trajectories.png'}")

    # Also plot top 20 countries by trajectory length for cleaner view
    traj_lengths = traj_df.groupby('country_code').size().sort_values(ascending=False)
    top_countries = traj_lengths.head(20).index.tolist()

    fig, ax = plt.subplots(figsize=(14, 10))

    for pos in positions:
        ax.scatter(pos[0], pos[1], c='lightgray', s=80, alpha=0.3, zorder=1)

    country_names = traj_df.drop_duplicates('country_code').set_index('country_code')['country_name'].to_dict()

    for i, country in enumerate(top_countries):
        country_data = traj_df[traj_df['country_code'] == country].sort_values('year')
        xs = country_data['bmu_x'].values
        ys = country_data['bmu_y'].values
        color = colors[i]

        ax.plot(xs, ys, '-', color=color, alpha=0.7, linewidth=2, zorder=2,
               label=country_names.get(country, country))
        ax.scatter(xs[0], ys[0], c=[color], s=100, marker='o',
                  edgecolors='black', linewidth=1.5, zorder=3)
        ax.scatter(xs[-1], ys[-1], c=[color], s=100, marker='s',
                  edgecolors='black', linewidth=1.5, zorder=3)

    ax.set_xlabel('SOM Grid X')
    ax.set_ylabel('SOM Grid Y')
    ax.set_title('Top 20 Countries by Data Coverage\nCircle=Start, Square=End')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "top20_trajectories.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plots_dir / 'top20_trajectories.png'}")


def main():
    print("=" * 70)
    print("   World Panel Data Demo: Country Economic Trajectories")
    print("=" * 70)
    print()

    # Paths
    data_path = Path(__file__).parent.parent.parent / "world_panel" / "world_panel_cleaned.csv"
    output_dir = Path(__file__).parent.parent / "output" / "world_panel"

    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("-" * 70)
    print("Step 1: Loading World Panel Data")
    print("-" * 70)

    df, feature_cols = load_and_prepare_data(data_path)

    # =========================================================================
    # Step 2: Create Sliding Windows (NO NORMALIZATION)
    # =========================================================================
    print("-" * 70)
    print("Step 2: Creating Sliding Windows (2 years, raw data)")
    print("-" * 70)

    X, entity_ids, time_indices, window_feature_names = create_sliding_windows(
        df, feature_cols, window_size=2
    )

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)
    print(f"\nFinal data shape: {X.shape}")

    # =========================================================================
    # Step 3: Train SE-DBGSOM
    # =========================================================================
    print("-" * 70)
    print("Step 3: Training SE-DBGSOM")
    print("-" * 70)

    som = SEDBGSOM(
        bayesian=True,
        bayesian_trials=25,
        bayesian_te_constraint=0.35,  # relaxed for high-dim data
        bayesian_ranges={
            'lambda_': (2, 10),
            'max_neurons': (250, 250),
            'n_iter': (100, 150),
            'init_size': (2, 2)
        },
        random_state=42
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
    # Step 4: Create Yearly Snapshots with Clustering
    # =========================================================================
    print("-" * 70)
    print("Step 4: Creating Yearly Snapshots & Clustering")
    print("-" * 70)

    traj_df = create_year_snapshots(
        som, X, entity_ids, time_indices, df, window_feature_names, output_dir
    )

    # =========================================================================
    # Step 5: Save All Trajectories
    # =========================================================================
    print("-" * 70)
    print("Step 5: Saving All Country Trajectories")
    print("-" * 70)

    raw_data_dir = output_dir / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    save_all_trajectories(traj_df, raw_data_dir)

    # Save neuron weights
    positions, weights = som.get_neuron_weights()
    neuron_data = []
    for i, pos in enumerate(positions):
        row = {'neuron_x': pos[0], 'neuron_y': pos[1]}
        for j, fname in enumerate(window_feature_names[:weights.shape[1]]):
            row[fname] = weights[i, j]
        row['hit_count'] = som.get_hit_counts(X).get(pos, 0)
        neuron_data.append(row)

    neuron_df = pd.DataFrame(neuron_data)
    neuron_df.to_csv(raw_data_dir / "neuron_weights.csv", index=False)
    print(f"Saved: {raw_data_dir / 'neuron_weights.csv'}")

    # =========================================================================
    # Step 6: Visualization
    # =========================================================================
    print("-" * 70)
    print("Step 6: Visualization")
    print("-" * 70)

    # Plot yearly snapshots (every 5 years)
    plot_yearly_snapshots(som, traj_df, output_dir)

    # Plot all trajectories
    plot_all_trajectories(som, traj_df, output_dir)

    print()
    print("=" * 70)
    print(f"Demo complete! Output saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
