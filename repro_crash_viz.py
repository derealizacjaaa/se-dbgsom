
print("Starting repro_crash_viz.py")
import numpy as np
import pandas as pd
from pathlib import Path

# Explicitly test matplotlib import first (sometimes interacts with Julia)
import matplotlib.pyplot as plt

print("Importing dbgsom...")
from dbgsom import SEDBGSOM
from dbgsom.clustering import SOMVesantoClustering
from dbgsom.visualization import DBGSOMVisualizer

def main():
    print("Entering main...")
    X = np.random.rand(50, 5)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    
    print("Initializing SEDBGSOM...")
    som = SEDBGSOM(
        lambda_=4, max_neurons=50, n_iter=10, init_size=(3, 3),
        random_state=42,
    )
    som.fit(df)
    print("Fit complete.")
    
    print("Clustering...")
    clustering = SOMVesantoClustering(method='ward', auto_k_method='davies_bouldin')
    labels = clustering.fit(som, df)
    print("Clustering complete.")
    
    print("Visualizing...")
    # Just init visualizer, don't necessarily plot everything if it saves files
    viz = DBGSOMVisualizer(Path("."), "repro_viz")
    print("Visualizer initialized.")
    # Try one plot function that uses matplotlib
    viz.plot_hit_counts(som, df, "repro_hits.png")
    print("Plot saved.")

if __name__ == "__main__":
    main()
