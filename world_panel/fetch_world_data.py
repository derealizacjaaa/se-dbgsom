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
    'BX.KLT.DINV.WD.GD.ZS': 'FDI_net_inflows_percent_of_GDP',
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
    
    # Process in chunks of 20 countries to avoid timeouts/URI length issues
    chunks = list(chunk_list(country_codes, 20))
    total_chunks = len(chunks)
    
    for i, batch in enumerate(chunks):
        country_str = ';'.join(batch)
        
        # Retry logic for the chunk
        max_retries = 3
        for attempt in range(max_retries):
            success = False
            page = 1
            chunk_data = [] # Store data for this chunk temporarily
            
            while True:
                # Fetch for specific countries
                url = f"{BASE_URL}/country/{country_str}/indicator/{code}?format=json&per_page=1000&date=1980:2023&page={page}"
                
                try:
                    response = requests.get(url, timeout=45) # Increased timeout
                    if response.status_code != 200:
                        # If 400, it might be a permanent error for this batch configuration, but could be transient
                        if response.status_code == 400:
                             print(f"  Error 400 for chunk {i+1}/{total_chunks}, page {page}. Retrying...")
                             time.sleep(2)
                             raise Exception("Error 400")
                        else:
                             print(f"  Error {response.status_code} for chunk {i+1}/{total_chunks}, page {page}. Retrying...")
                             time.sleep(2)
                             raise Exception(f"Error {response.status_code}")
                             
                    content = response.json()
                    
                    if not isinstance(content, list) or len(content) < 2:
                        success = True # access, end of pages
                        break
                        
                    metadata = content[0]
                    records = content[1]
                    
                    if not records:
                        success = True
                        break
                        
                    for item in records:
                        if item['value'] is not None:
                            chunk_data.append({
                                'Country Code': item['country']['id'],
                                'Country Name': item['country']['value'],
                                'Year': int(item['date']),
                                name: float(item['value'])
                            })
                    
                    if page >= metadata['pages']:
                        success = True
                        break
                        
                    page += 1
                    
                except Exception as e:
                    print(f"  Exception for chunk {i+1} (Attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(2 * (attempt + 1)) # Backoff
                    break # Break inner while loop to trigger retry of the chunk
            
            if success:
                data_list.extend(chunk_data)
                break # Move to next chunk
        
        time.sleep(0.5) # Delay between chunks
            
    return pd.DataFrame(data_list)

import concurrent.futures

def fetch_indicator_wrapper(args):
    code, name, countries = args
    return fetch_indicator_for_countries(code, name, countries)

def main():
    countries = fetch_countries()
    if not countries:
        print("Could not fetch countries.")
        return

    merged_df = None
    results = []
    
    # Use ThreadPoolExecutor to fetch indicators in parallel
    print(f"Fetching {len(INDICATORS)} indicators in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_indicator_wrapper, (code, name, countries)) for code, name in INDICATORS.items()]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                df_ind = future.result()
                if not df_ind.empty:
                    results.append(df_ind)
                else:
                    print(f"Warning: No data returned for an indicator.")
            except Exception as e:
                print(f"An error occurred during fetching: {e}")

    print("Merging datasets...")
    for df_ind in results:
        if merged_df is None:
            merged_df = df_ind
        else:
            merged_df = pd.merge(merged_df, df_ind, on=['Country Code', 'Country Name', 'Year'], how='outer')
            
    if merged_df is not None:
        merged_df.sort_values(by=['Country Name', 'Year'], inplace=True)
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(output_dir, 'world_panel_1980_2023.csv')
        
        print(f"Saving dataset to {output_file}...")
        merged_df.to_csv(output_file, index=False)
        print(f"Done! Shape: {merged_df.shape}")
    else:
        print("Failed to download data.")

if __name__ == "__main__":
    main()
