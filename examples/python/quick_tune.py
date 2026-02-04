import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dbgsom import SEDBGSOM

# Path setup
project_root = Path(os.getcwd())
sys.path.insert(0, str(project_root / "python"))
sys.path.append(str(project_root / "examples" / "python"))

try:
    from world_panel_vesanto_demo import (
        load_and_prepare_data, create_sliding_windows, 
        _cluster_active_neurons, FEATURE_COLS, 
        WINDOW_SIZE, STEP_SIZE, DECADE_YEARS, SNAPSHOT_YEARS
    )
except ImportError:
    print("Import failed")
    sys.exit(1)

def run():
    print("Loading data...")
    data_path = project_root / "world_panel" / "world_panel_som_ready.csv"
    df, feature_cols = load_and_prepare_data(data_path, year_start=1980)
    X, entity_ids, time_indices, _ = create_sliding_windows(
        df, feature_cols, window_size=WINDOW_SIZE, step_size=STEP_SIZE
    )

    # Targeted Grid
    # We want to lower k from ~11 to ~8.
    # Hypothesis: Higher lambda -> fewer neurons -> fewer clusters.
    lambdas = [3.0, 4.0, 5.0, 6.0]
    max_neurons_list = [50, 60, 70]
    
    results = []
    
    for lam in lambdas:
        for mn in max_neurons_list:
            print(f"Testing L={lam}, MN={mn}...")
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
            
            # Reconstruct minimalist dataframe for indexing
            full_df = pd.DataFrame({'time_index': time_indices, 'window_idx': range(len(time_indices))})
            
            decade_ks = []
            decade_dbs = []
            decade_sils = []
            
            for decade in DECADE_YEARS:
                next_decade = decade + 10
                decade_snapshots = [y for y in SNAPSHOT_YEARS if decade <= y < next_decade]
                
                # Gather BMUs
                decade_bmus = []
                for sy in decade_snapshots:
                    end_year = sy + WINDOW_SIZE - 1
                    mask = full_df['time_index'] == end_year
                    idxs = full_df[mask]['window_idx'].values
                    decade_bmus.extend([bmus[i] for i in idxs])
                
                # Cluster
                _, k, db_scores, sil_scores = _cluster_active_neurons(
                    decade_bmus, positions, weights, pos_to_idx,
                    min_clusters=4, max_clusters=15
                )
                
                decade_ks.append(k)
                decade_dbs.append(db_scores.get(k, np.nan))
                decade_sils.append(sil_scores.get(k, np.nan))
            
            avg_k = np.mean(decade_ks)
            avg_db = np.nanmean(decade_dbs)
            avg_sil = np.nanmean(decade_sils)
            
            res = {
                'lambda': lam,
                'max_neurons': mn,
                'avg_k': avg_k,
                'avg_db': avg_db,
                'avg_sil': avg_sil
            }
            results.append(res)
            print(f"  -> Avg k={avg_k:.1f}, DB={avg_db:.3f}, Sil={avg_sil:.3f}")

    # Save to file
    out_file = project_root / "tune_results.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    run()
