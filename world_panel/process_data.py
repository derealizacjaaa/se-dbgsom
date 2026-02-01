import pandas as pd
import os
import numpy as np

def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'world_panel_1990_2023.csv')
    output_file = os.path.join(base_dir, 'world_panel_cleaned.csv')
    
    print("Running World Panel Data Pipeline...")
    print("-" * 50)
    
    # 1. Load Raw Data
    if not os.path.exists(input_file):
        print(f"Error: Raw file not found: {input_file}")
        print("Please run 'fetch_world_data.py' first.")
        return

    print(f"Loading raw data: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Initial Shape: {df.shape}")
    
    # 2. Filter Countries (>30% missing data)
    print("\nStep 1: Filtering countries (>30% missing data)...")
    data_cols = [c for c in df.columns if c not in ['Country Code', 'Country Name', 'Year']]
    
    country_missing = df.groupby('Country Name')[data_cols].apply(lambda x: x.isnull().sum().sum())
    total_cells = df.groupby('Country Name')[data_cols].apply(lambda x: x.size)
    missing_pct = (country_missing / total_cells) * 100
    
    countries_to_keep = missing_pct[missing_pct <= 30].index.tolist()
    df_cleaned = df[df['Country Name'].isin(countries_to_keep)].copy()
    
    dropped_count = df['Country Name'].nunique() - df_cleaned['Country Name'].nunique()
    print(f"  Dropped {dropped_count} countries.")
    print(f"  Remaining Countries: {df_cleaned['Country Name'].nunique()}")
    
    # 3. Mask R&D Data (Before 2010)
    print("\nStep 2: Masking R&D Expenditure (1990-2009)...")
    rnd_col = 'RD_expenditure_percent_of_GDP'
    if rnd_col in df_cleaned.columns:
        mask = df_cleaned['Year'] < 2010
        pre_mask_count = df_cleaned[rnd_col].count()
        df_cleaned.loc[mask, rnd_col] = np.nan
        post_mask_count = df_cleaned[rnd_col].count()
        print(f"  Masked {pre_mask_count - post_mask_count} values.")
        print(f"  Valid R&D values remaining: {post_mask_count}")
    else:
        print(f"  Warning: {rnd_col} not found in dataset.")

    # 4. Save Result
    print(f"\nSaving cleaned dataset to: {output_file}")
    df_cleaned.to_csv(output_file, index=False)
    print("Pipeline completed successfully.")
    print("-" * 50)

if __name__ == "__main__":
    run_pipeline()
