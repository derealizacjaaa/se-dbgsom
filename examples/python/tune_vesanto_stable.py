import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add python folder to sys.path to access dbgsom module
project_root = Path(os.getcwd())
sys.path.insert(0, str(project_root / "python"))

# Add examples/python to path for module import
sys.path.append(str(project_root / "examples" / "python"))

from dbgsom import SEDBGSOM

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
    sys.path.append(str(Path(__file__).parent))
    from world_panel_vesanto_demo import (
        load_and_prepare_data, 
        create_sliding_windows, 
        _cluster_active_neurons,
        FEATURE_COLS, WINDOW_SIZE, STEP_SIZE,
        DECADE_YEARS, SNAPSHOT_YEARS
    )

def tune_stable_params():
    # Setup data
    base_dir = Path(__file__).parent.parent.parent
    data_path = base_dir / "world_panel" / "world_panel_som_ready.csv"
    
    print("Loading data...")
    df, feature_cols = load_and_prepare_data(data_path, year_start=1980)
    X, entity_ids, time_indices, window_feature_names = create_sliding_windows(
        df, feature_cols, window_size=WINDOW_SIZE, step_size=STEP_SIZE
    )
    
    # ---------------------------------------------------------
    # Strict 8-10 Clusters Grid
    # ---------------------------------------------------------
    lambdas = [1.0]
    max_neurons_list = [50, 60, 70, 80, 90, 100]
    
    results = []

    print("\nStarting Strict 8-10 Cluster Search...")
    print("Constraint: 8 <= k <= 10. Optimizing for decent DB score.")
    print("-" * 80)
    print(f"{'Lambda':<8} {'MaxN':<8} {'Decade':<8} {'k':<5} {'DB':<8} {'Sil':<8}")
    print("-" * 80)

    for lam in lambdas:
        for mn in max_neurons_list:
            # Train SOM
            som = SEDBGSOM(
                lambda_=lam,
                max_neurons=mn,
                n_iter=60, 
                init_size=(2, 2),
                random_state=42
            )
            som.fit(X)
            
            bmus = som.predict(X)
            positions, weights = som.get_neuron_weights()
            pos_to_idx = {pos: i for i, pos in enumerate(positions)}
            
            full_df = pd.DataFrame({
                'time_index': time_indices,
                'window_idx': range(len(time_indices))
            })

            # Evaluate each decade
            decade_metrics = []
            ks = []
            
            for decade in DECADE_YEARS:
                next_decade = decade + 10
                decade_snapshot_years = [y for y in SNAPSHOT_YEARS if decade <= y < next_decade]
                
                all_decade_bmus = []
                for sy in decade_snapshot_years:
                    end_year = sy + WINDOW_SIZE - 1
                    mask = full_df['time_index'] == end_year
                    idxs = full_df[mask]['window_idx'].values
                    all_decade_bmus.extend([bmus[i] for i in idxs])
                
                # STRICT CONSTRAINT: min_clusters=8, max_clusters=10
                labels, k, db_scores, sil_scores = _cluster_active_neurons(
                    all_decade_bmus, positions, weights, pos_to_idx,
                    min_clusters=8, max_clusters=10 
                )
                
                db = db_scores.get(k, np.nan)
                sil = sil_scores.get(k, np.nan)
                
                decade_metrics.append({
                    'decade': decade,
                    'k': k,
                    'db': db,
                    'sil': sil
                })
                ks.append(k)
                
                # print(f"{lam:<8} {mn:<8} {decade:<8} {k:<5} {db:.3f}    {sil:.3f}")

            # Criteria:
            # 1. All ks in [8, 10] (already enforced by _cluster_active_neurons logic)
            # 2. Minimize Avg DB score
            
            avg_k = np.mean(ks)
            k_std = np.std(ks)
            avg_db = np.mean([m['db'] for m in decade_metrics])
            avg_sil = np.mean([m['sil'] for m in decade_metrics])
            
            in_target_range = sum(1 for k in ks if 8 <= k <= 10)
            
            print(f"L={lam:<5} N={mn:<5} | k_range={min(ks)}-{max(ks)} | avg_k={avg_k:.1f} | Avg DB={avg_db:.3f}")

            results.append({
                'lambda': lam,
                'max_neurons': mn,
                'ks': ks,
                'avg_k': avg_k,
                'k_std': k_std,
                'avg_db': avg_db,
                'avg_sil': avg_sil,
                'in_target': in_target_range
            })
            
    print("\n" + "="*80)
    print("Top Configurations for Stable 8-10 Clusters:")
    print("="*80)
    
    # Sort by:
    # 1. Number of decades in target range (descending)
    # 2. Standard deviation of k (ascending)
    # 3. Average Silhouette (descending)
    
    results.sort(key=lambda x: (-x['in_target'], x['k_std'], -x['avg_sil']))
    
    for r in results[:10]:
        print(f"Lambda={r['lambda']}, MaxN={r['max_neurons']}")
        print(f"  ks per decade: {r['ks']}")
        print(f"  Avg k: {r['avg_k']:.2f}, Std k: {r['k_std']:.2f}")
        print(f"  Decades in [8,10]: {r['in_target']}/{len(DECADE_YEARS)}")
        print(f"  Avg Silhouette: {r['avg_sil']:.3f}")
        print("-" * 40)

    # Save to CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv("tune_stable_results.csv", index=False)
    print("Saved full results to tune_stable_results.csv")

if __name__ == "__main__":
    tune_stable_params()
