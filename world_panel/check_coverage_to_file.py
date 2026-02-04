import pandas as pd
import requests
import time
import random

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
    url = f"{BASE_URL}/country?format=json&per_page=300"
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        if len(data) < 2: return []
        countries = [item['id'] for item in data[1] if item['region']['value'] != 'Aggregates']
        return countries
    except: return []

def main():
    countries = fetch_countries()
    if not countries: return

    random.seed(42) 
    sample_countries = random.sample(countries, min(len(countries), 20))
    years_to_check = [1960, 1970, 1975, 1980, 1985, 1989]
    country_str = ';'.join(sample_countries)

    with open('world_panel/investigation_report.txt', 'w') as f:
        f.write(f"Indicator | 1960 | 1970 | 1975 | 1980 | 1985 | 1989\n")
        f.write("-" * 100 + "\n")
        
        for code, name in INDICATORS.items():
            url = f"{BASE_URL}/country/{country_str}/indicator/{code}?format=json&per_page=1000&date=1960:1989"
            counts = {y: 0 for y in years_to_check}
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    content = response.json()
                    if len(content) > 1 and content[1]:
                        for item in content[1]:
                            if item['value'] is not None:
                                try:
                                    y = int(item['date'])
                                    if y in counts:
                                        counts[y] += 1
                                except: pass
            except Exception: pass

            row = f"{name:<50}"
            for y in years_to_check:
                pct = counts[y] / len(sample_countries)
                row += f" | {pct:.0%}"
            f.write(row + "\n")
            print(f"Processed {name}")

if __name__ == "__main__":
    main()
