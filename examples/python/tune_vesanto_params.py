import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dbgsom import SEDBGSOM

import os

# Add python folder to sys.path to access dbgsom module
# Asuming we run from project root: python examples/python/tune_vesanto_params.py
project_root = Path(os.getcwd())
sys.path.insert(0, str(project_root / "python"))

# Add examples/python to path for module import
sys.path.append(str(project_root / "examples" / "python"))

# Import helpers from the demo script
try:
    from world_panel_vesanto_demo import (
        load_and_prepare_data, 
        create_sliding_windows, 
        _cluster_active_neurons,
        FEATURE_COLS, WINDOW_SIZE, STEP_SIZE,
        DECADE_YEARS, SNAPSHOT_YEARS
    )
except ImportError:
    # If straightforward import fails due to path issues, append local dir
    sys.path.append(str(Path(__file__).parent))
    from world_panel_vesanto_demo import (
        load_and_prepare_data, 
        create_sliding_windows, 
        _cluster_active_neurons,
        FEATURE_COLS, WINDOW_SIZE, STEP_SIZE,
        DECADE_YEARS, SNAPSHOT_YEARS
    )

def tune_params():
    # Setup data
    base_dir = Path(__file__).parent.parent.parent
    data_path = base_dir / "world_panel" / "world_panel_som_ready.csv"
    
    print("Loading data...")
    df, feature_cols = load_and_prepare_data(data_path, year_start=1980)
    X, entity_ids, time_indices, window_feature_names = create_sliding_windows(
        df, feature_cols, window_size=WINDOW_SIZE, step_size=STEP_SIZE
    )
    
    # Parameter grid
    lambdas = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    max_neurons_list = [40, 60, 80, 100]
    
    results = []

    print("\nStarting Grid Search...")
    print(f"{'Lambda':<8} {'MaxN':<8} {'Decade':<8} {'k':<5} {'DB':<8} {'Sil':<8}")
    print("-" * 50)

    for lam in lambdas:
        for mn in max_neurons_list:
            # Train SOM
            som = SEDBGSOM(
                lambda_=lam,
                max_neurons=mn,
                n_iter=50, # Reduced iterations for speed during tuning
                init_size=(2, 2),
                random_state=42
            )
            som.fit(X)
            
            bmus = som.predict(X)
            positions, weights = som.get_neuron_weights()
            pos_to_idx = {pos: i for i, pos in enumerate(positions)}
            
            # Setup for full_df (reuse logic to find decade indices)
            # We can simplify by just getting "active neurons for decade" roughly
            # But adhering to the demo's exact logic ensures consistency.
            
            # Reconstruct full_df minimalist version for index lookup
            # We only need 'time_index' and 'window_idx'
            full_df = pd.DataFrame({
                'time_index': time_indices,
                'window_idx': range(len(time_indices))
            })

            # Evaluate each decade
            decade_metrics = []
            
            for decade in DECADE_YEARS:
                next_decade = decade + 10
                decade_snapshot_years = [y for y in SNAPSHOT_YEARS if decade <= y < next_decade]
                
                all_decade_bmus = []
                for sy in decade_snapshot_years:
                    end_year = sy + WINDOW_SIZE - 1
                    mask = full_df['time_index'] == end_year
                    idxs = full_df[mask]['window_idx'].values
                    all_decade_bmus.extend([bmus[i] for i in idxs])
                
                labels, k, db_scores, sil_scores = _cluster_active_neurons(
                    all_decade_bmus, positions, weights, pos_to_idx,
                    min_clusters=4, max_clusters=15 
                )
                
                db = db_scores.get(k, np.nan)
                sil = sil_scores.get(k, np.nan)
                
                decade_metrics.append({
                    'decade': decade,
                    'k': k,
                    'db': db,
                    'sil': sil
                })
                
                print(f"{lam:<8} {mn:<8} {decade:<8} {k:<5} {db:.3f}    {sil:.3f}")

            # Compute average score for this config
            avg_k = np.mean([m['k'] for m in decade_metrics])
            avg_db = np.mean([m['db'] for m in decade_metrics])
            avg_sil = np.mean([m['sil'] for m in decade_metrics])
            
            results.append({
                'lambda': lam,
                'max_neurons': mn,
                'avg_k': avg_k,
                'avg_db': avg_db,
                'avg_sil': avg_sil,
                'decade_details': decade_metrics
            })
            
    print("\nTop 5 Configurations (closest to k=8, sorted by Silhouette):")
    # Sort by distance to k=8 then descending Silhouette
    # We want |avg_k - 8| minimized, then max avg_sil
    
    results.sort(key=lambda x: (abs(x['avg_k'] - 8), -x['avg_sil']))
    
    for r in results[:5]:
        print(f"Lambda={r['lambda']}, MaxN={r['max_neurons']} -> Avg k={r['avg_k']:.1f}, Avg DB={r['avg_db']:.3f}, Avg Sil={r['avg_sil']:.3f}")

if __name__ == "__main__":
    tune_params()
