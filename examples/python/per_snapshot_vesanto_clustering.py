"""
Per-Snapshot Vesanto Clustering with fixed k and full-grid propagation.

For each of the 9 snapshot years:
1. Cluster active SOM neurons via Ward linkage with a fixed k.
2. Propagate cluster labels to inactive neurons (nearest active neighbor
   in weight space).

Every neuron receives a cluster label for every snapshot, enabling
full-grid coloring in the visualization.

Reads:  raw_data/neuron_weights.csv, raw_data/all_trajectories.csv
Writes: raw_data/per_snapshot_clusters.csv

Run from project root:
    python examples/python/per_snapshot_vesanto_clustering.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, cdist
from sklearn.metrics import davies_bouldin_score


FIXED_K = {
    1990: 12, 1994: 12, 1998: 12,
    2002: 15, 2006: 15, 2010: 15,
    2014: 14,
    2018: 10, 2020: 10,
}


def main():
    output_dir = Path(__file__).parent.parent / "output" / "world_panel_vesanto"
    raw_dir = output_dir / "raw_data"

    neurons_df = pd.read_csv(raw_dir / "neuron_weights.csv")
    traj_df = pd.read_csv(raw_dir / "all_trajectories.csv")

    feat_cols = [c for c in neurons_df.columns
                 if c not in ("neuron_x", "neuron_y", "hit_count")]

    all_positions = []
    neuron_weights = {}
    for _, row in neurons_df.iterrows():
        key = (int(row["neuron_x"]), int(row["neuron_y"]))
        neuron_weights[key] = row[feat_cols].values.astype(float)
        all_positions.append(key)

    years = sorted(traj_df["snapshot_year"].unique())
    results = []

    print("Per-Snapshot Vesanto Clustering (fixed k, full-grid)")
    print("=" * 55)

    for year in years:
        snap = traj_df[traj_df["snapshot_year"] == year]
        active_set = set(
            zip(snap["bmu_x"].astype(int), snap["bmu_y"].astype(int))
        )
        active_pos = sorted(active_set)

        k = FIXED_K.get(year, 10)
        n_active = len(active_pos)

        # Cluster active neurons with Ward linkage at fixed k
        active_weights = np.array([neuron_weights[p] for p in active_pos])
        effective_k = min(k, n_active - 1)

        distances = pdist(active_weights, metric="euclidean")
        Z = linkage(distances, method="ward")
        active_labels = fcluster(Z, effective_k, criterion="maxclust")

        db = davies_bouldin_score(active_weights, active_labels)

        # Build lookup: active position -> cluster label
        active_label_map = {
            pos: int(active_labels[i])
            for i, pos in enumerate(active_pos)
        }

        # Propagate to inactive neurons: nearest active neighbor in weight space
        inactive_pos = [p for p in all_positions if p not in active_set]

        if inactive_pos:
            inactive_weights = np.array([neuron_weights[p] for p in inactive_pos])
            dists = cdist(inactive_weights, active_weights, metric="euclidean")
            nearest_idx = dists.argmin(axis=1)

            inactive_label_map = {
                pos: int(active_labels[nearest_idx[i]])
                for i, pos in enumerate(inactive_pos)
            }
        else:
            inactive_label_map = {}

        print(f"  {year}:  k={effective_k:2d}   DB={db:.3f}   "
              f"active={n_active}   total={len(all_positions)}")

        # Emit rows for ALL neurons
        for pos in all_positions:
            is_active = pos in active_set
            cl = active_label_map.get(pos) or inactive_label_map.get(pos, 1)
            results.append({
                "snapshot_year": year,
                "neuron_x": pos[0],
                "neuron_y": pos[1],
                "ps_cluster": cl,
                "optimal_k": effective_k,
                "db_score": round(db, 4),
                "is_active": int(is_active),
            })

    df = pd.DataFrame(results)
    out_path = raw_dir / "per_snapshot_clusters.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
