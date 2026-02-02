"""
Analyze Country Trajectories for Economic Interpretation.

Reads:
    - examples/output/world_panel_vesanto/raw_data/cluster_names.csv
    - examples/output/world_panel_vesanto/raw_data/all_trajectories.csv
Writes:
    - examples/output/world_panel_vesanto/country_case_studies.md
"""

import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).parent.parent / "output" / "world_panel_vesanto"
    raw_data_dir = base_dir / "raw_data"
    output_report = base_dir / "country_case_studies.md"
    
    # Load data
    cluster_names = pd.read_csv(raw_data_dir / "cluster_names.csv")
    trajectories = pd.read_csv(raw_data_dir / "all_trajectories.csv")
    
    # Selected countries for case study
    countries_of_interest = {
        'US': 'United States',
        'CN': 'China',
        'PL': 'Poland',
        'NG': 'Nigeria'
    }
    
    # --- Report Generation ---
    with open(output_report, "w") as f:
        f.write("# Country Trajectory Case Studies\n\n")
        f.write("Analysis of clustering trajectories for selected countries to verify economic interpretation.\n\n")
        
        for code, name in countries_of_interest.items():
            f.write(f"## {name} ({code})\n\n")
            
            country_data = trajectories[trajectories['country_code'] == code].sort_values('snapshot_year')
            
            if country_data.empty:
                f.write(f"> [!WARNING]\n> No data found for {name}.\n\n")
                continue
                
            f.write("| Year | Decade | Cluster ID | Cluster Name | GDP Centroid |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            
            for _, row in country_data.iterrows():
                decade = row['decade']
                cluster_id = row['cluster']
                
                # Find cluster name info
                cluster_info = cluster_names[
                    (cluster_names['decade'] == decade) & 
                    (cluster_names['cluster_id'] == cluster_id)
                ]
                
                if not cluster_info.empty:
                    c_name = cluster_info.iloc[0]['cluster_name']
                    c_gdp = cluster_info.iloc[0]['gdp_centroid']
                else:
                    c_name = "Unknown"
                    c_gdp = "N/A"
                
                f.write(f"| {row['snapshot_year']} | {decade}s | {cluster_id} | {c_name} | {c_gdp} |\n")
            
            f.write("\n")
            
            # Brief Summary Logic (Basic heuristics)
            start_cluster = country_data.iloc[0]['cluster']
            end_cluster = country_data.iloc[-1]['cluster']
            
            # Get names for start/end
            start_name = cluster_names[(cluster_names['decade'] == country_data.iloc[0]['decade']) & (cluster_names['cluster_id'] == start_cluster)]['cluster_name'].values[0] if not cluster_names.empty else "Unknown"
            end_name = cluster_names[(cluster_names['decade'] == country_data.iloc[-1]['decade']) & (cluster_names['cluster_id'] == end_cluster)]['cluster_name'].values[0] if not cluster_names.empty else "Unknown"
            
            f.write(f"**Trajectory Summary**: Started in **{start_name}** and ended in **{end_name}**.\n\n")

    print(f"Report generated: {output_report}")

if __name__ == "__main__":
    main()
