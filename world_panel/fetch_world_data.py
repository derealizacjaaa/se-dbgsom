import pandas as pd
import requests
import time
import os

# Define the list of variables (Code: Name)
INDICATORS = {
    'NY.GDP.PCAP.KD': 'GDP_per_capita_constant_2015_US',
    'NE.TRD.GNFS.ZS': 'Trade_percent_of_GDP',
    'SL.UEM.TOTL.ZS': 'Unemployment_total_percent_of_labor_force',
    'NV.AGR.TOTL.ZS': 'Agriculture_value_added_percent_of_GDP',
    'NV.IND.TOTL.ZS': 'Industry_value_added_percent_of_GDP',
    'SP.URB.TOTL.IN.ZS': 'Urban_population_percent',
    'FP.CPI.TOTL.ZG': 'Inflation_consumer_prices_annual_percent',
    'NE.GDI.FTOT.ZS': 'Gross_fixed_capital_formation_percent_of_GDP',
    'EG.ELC.ACCS.ZS': 'Access_to_electricity_percent_of_population',
    'IT.NET.USER.ZS': 'Internet_users_percent_of_population',
    'NV.SRV.TOTL.ZS': 'Services_value_added_percent_of_GDP',
    'NY.GDS.TOTL.ZS': 'Gross_domestic_savings_percent_of_GDP',
    'SE.XPD.TOTL.GD.ZS': 'Gov_expenditure_on_education_percent_of_GDP',
    'SE.PRM.ENRR': 'Primary_school_enrollment_gross_percent',
    'GB.XPD.RSDV.GD.ZS': 'RD_expenditure_percent_of_GDP',
    'FS.AST.PRVT.GD.ZS': 'Domestic_credit_to_private_sector_percent_of_GDP',
    'SP.DYN.TFRT.IN': 'Fertility_rate_total',
    'BX.KLT.DINV.WD.GD.ZS': 'FDI_net_inflows_percent_of_GDP',
    'SP.POP.TOTL': 'Population_total',
    'AG.LND.TOTL.K2': 'Land_area_sq_km'
}

BASE_URL = "https://api.worldbank.org/v2"

def fetch_countries():
    print("Fetching list of countries...")
    url = f"{BASE_URL}/country?format=json&per_page=300"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if len(data) < 2:
            return []
        
        countries = [item['id'] for item in data[1] if item['region']['value'] != 'Aggregates']
        print(f"Found {len(countries)} countries (excluding aggregates).")
        return countries
    except Exception as e:
        print(f"Error fetching countries: {e}")
        return []

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def fetch_indicator_for_countries(code, name, country_codes):
    print(f"Fetching {name} ({code})...")
    data_list = []
    
    # Process in chunks of 50 countries (API allows semicolon separated)
    chunks = list(chunk_list(country_codes, 50))
    total_chunks = len(chunks)
    
    for i, batch in enumerate(chunks):
        country_str = ';'.join(batch)
        page = 1
        
        while True:
            # Fetch for specific countries
            url = f"{BASE_URL}/country/{country_str}/indicator/{code}?format=json&per_page=1000&date=1990:2023&page={page}"
            
            try:
                response = requests.get(url, timeout=30)
                if response.status_code != 200:
                    print(f"  Error {response.status_code} for chunk {i+1}/{total_chunks}, page {page}")
                    time.sleep(2)
                    continue
                    
                content = response.json()
                
                if not isinstance(content, list) or len(content) < 2:
                    break
                    
                metadata = content[0]
                records = content[1]
                
                if not records:
                    break
                    
                for item in records:
                    if item['value'] is not None:
                        data_list.append({
                            'Country Code': item['country']['id'],
                            'Country Name': item['country']['value'],
                            'Year': int(item['date']),
                            name: float(item['value'])
                        })
                
                if page >= metadata['pages']:
                    break
                    
                page += 1
                
            except Exception as e:
                print(f"  Exception for chunk {i+1}: {e}")
                time.sleep(2)
                break
        
        time.sleep(0.2) # Small delay between chunks
            
    return pd.DataFrame(data_list)

def main():
    countries = fetch_countries()
    if not countries:
        print("Could not fetch countries.")
        return

    merged_df = None
    
    for code, name in INDICATORS.items():
        # Retry logic for the whole indicator if needed, but chunking is usually robust
        df_ind = fetch_indicator_for_countries(code, name, countries)
        
        if df_ind.empty:
            print(f"Warning: No data found for {name}")
            continue
            
        if merged_df is None:
            merged_df = df_ind
        else:
            merged_df = pd.merge(merged_df, df_ind, on=['Country Code', 'Country Name', 'Year'], how='outer')
            
    if merged_df is not None:
        merged_df.sort_values(by=['Country Name', 'Year'], inplace=True)
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(output_dir, 'world_panel_1990_2023.csv')
        
        print(f"Saving dataset to {output_file}...")
        merged_df.to_csv(output_file, index=False)
        print(f"Done! Shape: {merged_df.shape}")
    else:
        print("Failed to download data.")

if __name__ == "__main__":
    main()
