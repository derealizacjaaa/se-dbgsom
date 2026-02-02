import pandas as pd
import os
import numpy as np


# ---------------------------------------------------------------------------
# Feature transformations for SOM-ready data
# ---------------------------------------------------------------------------

def _log_rescale(series, target_min=0, target_max=100):
    """
    ln-transform a strictly-positive series, then min-max rescale.

    Useful for log-normally distributed features like GDP per capita where
    equal *ratios* should map to equal *differences* in the SOM.
    """
    valid = series.dropna()
    if valid.empty or (valid <= 0).any():
        return series

    log_vals = np.log(series)
    log_min = np.log(valid.min())
    log_max = np.log(valid.max())

    if log_max == log_min:
        return series

    return (log_vals - log_min) / (log_max - log_min) * (target_max - target_min) + target_min


def _signed_log_rescale(series, target_min=0, target_max=100):
    """
    Signed-log transform: sign(x) * ln(1 + |x|), then min-max rescale.

    Handles features with negative values and extreme positive outliers
    (e.g. inflation with hyperinflation episodes).
    """
    valid = series.dropna()
    if valid.empty:
        return series

    signed_log = np.sign(series) * np.log1p(np.abs(series))
    sl_valid = signed_log.dropna()
    sl_min = sl_valid.min()
    sl_max = sl_valid.max()

    if sl_max == sl_min:
        return series

    return (signed_log - sl_min) / (sl_max - sl_min) * (target_max - target_min) + target_min


def _minmax_rescale(series, target_min=0, target_max=100):
    """
    Simple min-max rescale for features with small natural ranges
    (e.g. fertility rate 0.6-8.6) that would otherwise be invisible.
    """
    valid = series.dropna()
    if valid.empty:
        return series

    v_min = valid.min()
    v_max = valid.max()

    if v_max == v_min:
        return series

    return (series - v_min) / (v_max - v_min) * (target_max - target_min) + target_min


def transform_features(df):
    """
    Apply SOM-appropriate transformations so no single feature dominates
    Euclidean distance. All transformed features land in [0, 100].

    Transforms applied:
      - GDP per capita:  ln + rescale  (log-normal distribution, 600x range)
      - Inflation:       signed-log + rescale  (hyperinflation outliers)
      - Fertility rate:  min-max rescale  (tiny natural range ~0.6-8.6)

    Percentage-of-GDP features (Trade, Agriculture, Industry, etc.) already
    sit in a 0-100-ish range and are left untouched.
    """
    df = df.copy()

    transforms = {
        'GDP_per_capita_constant_2015_US': ('log_rescale', lambda s: _log_rescale(s, 0, 200)),
        'Inflation_consumer_prices_annual_percent': ('signed_log', _signed_log_rescale),
        'Fertility_rate_total': ('minmax', _minmax_rescale),
    }

    for col, (label, fn) in transforms.items():
        if col not in df.columns:
            continue

        valid_before = df[col].dropna()
        df[col] = fn(df[col])
        valid_after = df[col].dropna()

        print(f"  {col}")
        print(f"    Transform: {label}")
        print(f"    Before: [{valid_before.min():.1f}, {valid_before.max():.1f}]  std={valid_before.std():.1f}")
        print(f"    After:  [{valid_after.min():.1f}, {valid_after.max():.1f}]  std={valid_after.std():.1f}")

    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'world_panel_1990_2023.csv')
    som_ready_file = os.path.join(base_dir, 'world_panel_som_ready.csv')

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

    # 4. Transform features for SOM training
    print("\nStep 3: Transforming features for SOM training...")
    df_som = transform_features(df_cleaned)

    print(f"\nSaving SOM-ready dataset: {som_ready_file}")
    df_som.to_csv(som_ready_file, index=False)

    print("\nPipeline completed successfully.")
    print("-" * 50)
    print(f"  SOM-ready (transformed): {som_ready_file}")


if __name__ == "__main__":
    run_pipeline()
