
print("Starting repro_crash_full.py")
import numpy as np
import pandas as pd
from pathlib import Path

print("Importing dbgsom...")
try:
    from dbgsom import (
        SEDBGSOM,
        assign_cluster_labels,
        compute_all_metrics,
    )
    print("dbgsom imported successfully.")
except Exception as e:
    print(f"Error importing dbgsom: {e}")
    exit(1)

from dbgsom.clustering import SOMVesantoClustering

def main():
    print("Entering main...")
    # Mock data
    X = np.random.rand(100, 5)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    
    print("Initializing SEDBGSOM...")
    try:
        som = SEDBGSOM(
            lambda_=4, max_neurons=140, n_iter=10, init_size=(3, 3),
            random_state=42,
        )
        print("SEDBGSOM initialized.")
        
        print("Fitting SEDBGSOM...")
        som.fit(df)
        print("Fit complete.")
        
        print("Final neurons:", som.n_neurons_)
        
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    main()
