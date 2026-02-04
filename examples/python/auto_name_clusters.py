
import pandas as pd
import os
import numpy as np

def auto_name_clusters():
    # Paths
    base_dir = r"c:\Users\maxok\Documents\dbgsom\examples\output\world_panel_vesanto\raw_data"
    traj_path = os.path.join(base_dir, "all_trajectories.csv")
    names_path = os.path.join(base_dir, "cluster_names.csv")
    
    # Load data
    try:
        traj_df = pd.read_csv(traj_path)
    except FileNotFoundError:
        print(f"Error: File not found at {traj_path}")
        return

    # Load existing names file to preserve structure
    try:
        names_df = pd.read_csv(names_path)
    except FileNotFoundError:
        print(f"Error: File not found at {names_path}")
        return

    # Prepare for updates
    new_names_map = {} # (decade, cluster_id) -> name

    # Process each decade
    for decade in sorted(traj_df['decade'].unique()):
        print(f"Processing Decade {decade}...")
        df_decade = traj_df[traj_df['decade'] == decade].copy()
        
        # Mapping columns
        cols_map = {
            'GDP_per_capita_constant_2015_US_t0': 'GDP_pc',
            'Agriculture_value_added_percent_of_GDP_t0': 'Agri_share',
            'Industry_value_added_percent_of_GDP_t0': 'Ind_share',
            'Services_value_added_percent_of_GDP_t0': 'Serv_share',
            'Urban_population_percent_t0': 'Urban_pop',
            'Trade_percent_of_GDP_t0': 'Trade_share'
        }
        df_decade.rename(columns=cols_map, inplace=True)
        # Force numeric
        df_decade['GDP_pc'] = pd.to_numeric(df_decade['GDP_pc'], errors='coerce')
        
        # Calculate cluster stats
        clusters = df_decade['cluster'].unique()
        cluster_stats = []
        
        for cluster_id in clusters:
            c_data = df_decade[df_decade['cluster'] == cluster_id]
            stats = {
                'id': cluster_id,
                'gdp': c_data['GDP_pc'].median(),
                'agri': c_data['Agri_share'].mean(),
                'ind': c_data['Ind_share'].mean(),
                'serv': c_data['Serv_share'].mean(),
                'trade': c_data['Trade_share'].mean(),
                'urban': c_data['Urban_pop'].mean()
            }
            cluster_stats.append(stats)
            
        # Sort by GDP to determine income levels
        cluster_stats.sort(key=lambda x: x['gdp'])
        
        # Determine names
        num_clusters = len(cluster_stats)
        for i, stat in enumerate(cluster_stats):
            name = "Unknown"
            
            # Structural Characteristics
            is_trade_hub = stat['trade'] > 150
            is_agrarian = stat['agri'] > 20
            is_industrial = stat['ind'] > 35
            is_service = stat['serv'] > 60
            is_very_industrial = stat['ind'] > 50
            
            # Income Level (Relative rank in decade)
            # Simple quantile-like logic
            percentile = (i + 1) / num_clusters
            
            if percentile <= 0.25:
                income_tag = "Developing"
            elif percentile <= 0.5:
                income_tag = "Emerging"
            elif percentile <= 0.8:
                income_tag = "Industrializing/Transitional" # Placeholder
            else:
                income_tag = "Advanced"

            # Construct Name
            if is_trade_hub:
                name = "Global Trade Hubs"
            elif is_very_industrial and stat['serv'] < 40:
                name = "Energy/Industrial Exporters"
            elif is_agrarian:
                name = f"Agrarian {income_tag}"
            elif is_service:
                if income_tag == "Advanced":
                    name = "Advanced Service Economies"
                else:
                    name = f"Service-based {income_tag}"
            elif is_industrial:
                if income_tag == "Advanced":
                     name = "High-Income Industrial"
                else:
                    name = f"Industrial {income_tag}"
            else:
                # Fallback based on simple income
                if income_tag == "Industrializing/Transitional":
                     name = "Transitional Economies"
                else:
                    name = f"{income_tag} Economies"

            # Refinements for clarity
            if name == "Agrarian Advanced": name = "Advanced Agrarian" # Unlikely but possible (NZ?)
            if name == "Agrarian Developing": name = "Agrarian Developing" # Good
            if name == "Agrarian Emerging": name = "Emerging Agrarian"
            
            # Specific fixes for known patterns
            # Extremely low GDP?
            if percentile <= 0.25 and not is_agrarian and not is_trade_hub:
                 name = "Fragile Economies"

            # Store result
            new_names_map[(decade, stat['id'])] = name
            print(f"  Cluster {stat['id']}: Rank={i}/{num_clusters}, P={percentile:.2f}, GDP={stat['gdp']:.0f} -> {name}")

    # Update the dataframe
    # Ensure cluster_id column name (handle 'cluster' vs 'cluster_id' mess if any, but previous step fixed it to cluster_id)
    # The file currently has columns: decade,cluster_id,cluster_name,gdp_centroid,n_neurons
    
    updated_rows = 0
    # Standardize column name if needed
    if 'cluster' in names_df.columns and 'cluster_id' not in names_df.columns:
        names_df.rename(columns={'cluster': 'cluster_id'}, inplace=True)
        
    for idx, row in names_df.iterrows():
        key = (row['decade'], row['cluster_id'])
        if key in new_names_map:
            names_df.at[idx, 'cluster_name'] = new_names_map[key]
            updated_rows += 1
            
    # Save
    names_df.to_csv(names_path, index=False)
    print(f"Update complete. {updated_rows} rows updated in {names_path}")

if __name__ == "__main__":
    auto_name_clusters()
