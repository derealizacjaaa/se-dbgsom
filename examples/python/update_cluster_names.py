
import pandas as pd
import os

def update_cluster_names():
    # Paths
    base_dir = r"c:\Users\maxok\Documents\dbgsom\examples\output\world_panel_vesanto\raw_data"
    csv_path = os.path.join(base_dir, "cluster_names.csv")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    # Load data
    df = pd.read_csv(csv_path)
    
    # Mapping of Cluster ID to New Name
    # Note: IDs in the dataframe might be integers or strings, we should handle both.
    name_map = {
        2: "Agrarian Developing",
        1: "Fragile Economies",
        4: "Emerging Markets",
        3: "Energy Exporters",
        6: "Service Transition",
        8: "Industrializing High-Income",
        5: "Global Trade Hubs",
        7: "Advanced Service Economies"
    }

    # Ensure cluster column is numeric if possible for matching
    # (It typically is, but let's be safe)
    try:
        df['cluster_id'] = pd.to_numeric(df['cluster_id'])
    except ValueError:
        print("Warning: 'cluster_id' column contains non-numeric values. proceeding with string matching if needed.")

    # Update logic
    # We iterate through the map and apply changes
    for cluster_id, new_name in name_map.items():
        # Mask for the specific cluster ID
        mask = (df['cluster_id'] == cluster_id)
        
        count = mask.sum()
        if count > 0:
            df.loc[mask, 'cluster_name'] = new_name
            print(f"Updated {count} rows for Cluster {cluster_id} to '{new_name}'")
        else:
            print(f"Warning: Cluster {cluster_id} not found in the file.")

    # Save back to CSV
    df.to_csv(csv_path, index=False)
    print(f"Successfully saved updated names to {csv_path}")

if __name__ == "__main__":
    update_cluster_names()
