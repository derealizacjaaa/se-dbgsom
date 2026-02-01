import pandas as pd
import os

# Variable descriptions (mapped from fetch_world_data.py context)
DESCRIPTIONS = {
    'Country Code': 'ISO 3-letter country code',
    'Country Name': 'Name of the country or region',
    'Year': 'Year of observation (1990-2023)',
    'GDP_per_capita_constant_2015_US': 'GDP per capita in constant 2015 US dollars. Indicator of average economic output per person.',
    'Trade_percent_of_GDP': 'Sum of exports and imports of goods and services measured as a share of gross domestic product.',
    'Unemployment_total_percent_of_labor_force': 'Unemployment refers to the share of the labor force that is without work but available for and seeking employment.',
    'Agriculture_value_added_percent_of_GDP': 'Agriculture corresponds to ISIC divisions 1-5 and includes forestry, hunting, and fishing, as well as cultivation of crops and livestock production.',
    'Industry_value_added_percent_of_GDP': 'Industry corresponds to ISIC divisions 10-45 and includes manufacturing (ISIC divisions 15-37).',
    'Services_value_added_percent_of_GDP': 'Services correspond to ISIC divisions 50-99 and include value added in wholesale and retail trade (including hotels and restaurants), transport, and government, financial, professional, and personal services.',
    'Urban_population_percent': 'Urban population refers to people living in urban areas as defined by national statistical offices.',
    'Inflation_consumer_prices_annual_percent': 'Inflation as measured by the consumer price index reflects the annual percentage change in the cost to the average consumer of acquiring a basket of goods and services.',
    'Gross_fixed_capital_formation_percent_of_GDP': 'Gross fixed capital formation (formerly gross domestic fixed investment) includes land improvements (fences, ditches, drains, and so on); plant, machinery, and equipment purchases; and the construction of roads, railways, and the like.',
    'Access_to_electricity_percent_of_population': 'Access to electricity is the percentage of population with access to electricity.',
    'Internet_users_percent_of_population': 'Internet users are individuals who have used the Internet (from any location) in the last 12 months. Internet can be used via a computer, mobile phone, personal digital assistant, games machine, digital TV etc.',
    'Gross_domestic_savings_percent_of_GDP': 'Gross domestic savings are calculated as GDP less final consumption expenditure (total consumption).',
    'Gov_expenditure_on_education_percent_of_GDP': 'General government expenditure on education (current, capital, and transfers) is expressed as a percentage of GDP.',
    'Primary_school_enrollment_gross_percent': 'Gross enrollment ratio is the ratio of total enrollment, regardless of age, to the population of the age group that officially corresponds to the level of education shown.',
    'RD_expenditure_percent_of_GDP': 'Gross domestic expenditures on research and development (R&D), expressed as a percent of GDP. They include both capital and current expenditures.',
    'Domestic_credit_to_private_sector_percent_of_GDP': 'Domestic credit to private sector refers to financial resources provided to the private sector by financial corporations.',
    'Fertility_rate_total': 'Total fertility rate represents the number of children that would be born to a woman if she were to live to the end of her childbearing years and bear children in accordance with age-specific fertility rates of the specified year.',
    'FDI_net_inflows_percent_of_GDP': 'Foreign direct investment are the net inflows of investment to acquire a lasting management interest (10 percent or more of voting stock) in an enterprise operating in an economy other than that of the investor.',
    'Population_total': 'Total population is based on the de facto definition of population, which counts all residents regardless of legal status or citizenship.',
    'Land_area_sq_km': 'Land area is a country\'s total area, excluding area under inland water bodies, national claims to continental shelf, and exclusive economic zones.'
}

def generate_report():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'world_panel_cleaned.csv')
    output_file = os.path.join(base_dir, 'DATA_DICTIONARY.md')
    
    print(f"Generating report from {input_file}...")
    df = pd.read_csv(input_file)
    
    with open(output_file, 'w') as f:
        f.write("# World Panel Dataset: Data Dictionary & Statistics\n\n")
        f.write("**File:** `world_panel_cleaned.csv`\n")
        f.write(f"**Dimensions:** {df.shape[0]} rows × {df.shape[1]} columns\n")
        f.write(f"**Period:** {df['Year'].min()} - {df['Year'].max()}\n")
        f.write(f"**Countries:** {df['Country Name'].nunique()}\n\n")
        
        f.write("## Variable Details\n\n")
        
        for col in df.columns:
            f.write(f"### {col}\n")
            desc = DESCRIPTIONS.get(col, "No description available.")
            f.write(f"**Description:** {desc}\n\n")
            
            if pd.api.types.is_numeric_dtype(df[col]) and col != 'Year':
                stats = df[col].describe()
                missing = df[col].isnull().sum()
                missing_pct = (missing / len(df)) * 100
                
                f.write("| Statistic | Value |\n")
                f.write("| :--- | :--- |\n")
                f.write(f"| Count | {int(stats['count'])} |\n")
                f.write(f"| Mean | {stats['mean']:.4f} |\n")
                f.write(f"| Std Dev | {stats['std']:.4f} |\n")
                f.write(f"| Min | {stats['min']:.4f} |\n")
                f.write(f"| 25% | {stats['25%']:.4f} |\n")
                f.write(f"| Median (50%) | {stats['50%']:.4f} |\n")
                f.write(f"| 75% | {stats['75%']:.4f} |\n")
                f.write(f"| Max | {stats['max']:.4f} |\n")
                f.write(f"| **Missing** | {missing} ({missing_pct:.2f}%) |\n")
            else:
                f.write(f"- **Type:** {df[col].dtype}\n")
                f.write(f"- **Unique Values:** {df[col].nunique()}\n")
                if col == 'Country Name':
                    f.write(f"- **Examples:** {', '.join(df[col].unique()[:5])}...\n")
            
            f.write("\n---\n\n")
            
    print(f"Report generated: {output_file}")

if __name__ == "__main__":
    generate_report()
