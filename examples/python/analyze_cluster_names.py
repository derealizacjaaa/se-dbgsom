
import pandas as pd
import os

def analyze_clusters():
    # Paths
    base_dir = r"c:\Users\maxok\Documents\dbgsom\examples\output\world_panel_vesanto\raw_data"
    traj_path = os.path.join(base_dir, "all_trajectories.csv")
    
    # Load data
    try:
        df = pd.read_csv(traj_path)
    except FileNotFoundError:
        print(f"Error: File not found at {traj_path}")
        return

    # Get all unique decades
    decades = sorted(df['decade'].unique())
    
    analysis_results = []

    for decade in decades:
        df_decade = df[df['decade'] == decade].copy()
        
        # Columns of interest (using t0 for current period)
        # Note: We need to check if columns exist, as names might vary slightly? 
        # Assuming consistent naming based on previous check, but 't0' suffix is likely consistent.
        cols_map = {
            'GDP_per_capita_constant_2015_US_t0': 'GDP_pc',
            'Agriculture_value_added_percent_of_GDP_t0': 'Agri_share',
            'Industry_value_added_percent_of_GDP_t0': 'Ind_share',
            'Services_value_added_percent_of_GDP_t0': 'Serv_share',
            'Urban_population_percent_t0': 'Urban_pop',
            'Trade_percent_of_GDP_t0': 'Trade_share',
            'Inflation_consumer_prices_annual_percent_t0': 'Inflation'
        }
        
        # Rename columns for simpler access
        df_decade.rename(columns=cols_map, inplace=True)
        
        clusters = df_decade['cluster'].unique()
        
        for cluster_id in clusters:
            cluster_data = df_decade[df_decade['cluster'] == cluster_id]
            
            # Calculate stats
            stats = {
                'Decade': decade,
                'Cluster': cluster_id,
                'Count': len(cluster_data),
                'Countries': ", ".join(sorted(cluster_data['country_name'].unique())),
                'Avg_GDP_pc': cluster_data['GDP_pc'].mean(),
                'Median_GDP_pc': cluster_data['GDP_pc'].median(),
                'Avg_Agri': cluster_data['Agri_share'].mean(),
                'Avg_Ind': cluster_data['Ind_share'].mean(),
                'Avg_Serv': cluster_data['Serv_share'].mean(),
                'Avg_Urban': cluster_data['Urban_pop'].mean(),
                'Avg_Trade': cluster_data['Trade_share'].mean()
            }
            analysis_results.append(stats)
        
    # Convert to DataFrame for sorting and printing
    results_df = pd.DataFrame(analysis_results)
    
    # Sort by Decade then Median GDP per capita
    results_df.sort_values(by=['Decade', 'Median_GDP_pc'], inplace=True)
    
    # Print formatted output
    print(f"{'Decade':<6} | {'ID':<4} | {'Count':<5} | {'GDP (Med)':<10} | {'Agri%':<6} | {'Ind%':<6} | {'Serv%':<6} | {'Urb%':<6} | {'Trade%':<6} | Countries")
    print("-" * 160)
    
    for _, row in results_df.iterrows():
        countries_list = row['Countries']
        if len(countries_list) > 80:
             countries_list = countries_list[:80] + "..."
             
        print(f"{int(row['Decade']):<6} | {int(row['Cluster']):<4} | {int(row['Count']):<5} | "
              f"{row['Median_GDP_pc']:<10.2f} | {row['Avg_Agri']:<6.1f} | {row['Avg_Ind']:<6.1f} | "
              f"{row['Avg_Serv']:<6.1f} | {row['Avg_Urban']:<6.1f} | {row['Avg_Trade']:<6.1f} | {countries_list}")

if __name__ == "__main__":
    analyze_clusters()
