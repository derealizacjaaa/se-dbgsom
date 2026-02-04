import requests
import pandas as pd

def get_microstates(threshold=1000000):
    url = f"http://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?date=2020&format=json&per_page=500"
    response = requests.get(url)
    data = response.json()
    
    if len(data) < 2:
        print("Error fetching data")
        return []

    microstates = []
    print(f"Checking {len(data[1])} countries/regions for Population < {threshold}...")
    
    for entry in data[1]:
        country = entry['country']
        value = entry['value']
        if value is None:
            continue
            
        if value < threshold:
            # Check if it's a valid country code (2 chars) not a region
            if len(entry['countryiso3code']) == 3: # API returns ISO3, but our dataset uses ISO2 usually? 
                # Wait, our dataset has 'Country Code'. Let's check format.
                # world_panel uses ISO2 (AF, AL, etc.)
                # API entry has 'id' which is ISO2 usually or ISO3?
                # entry['country']['id'] is ISO2.
                # entry['countryiso3code'] is ISO3.
                
                code = entry['country']['id']
                name = entry['country']['value']
                print(f"  {code} - {name}: {int(value):,}")
                microstates.append(code)
                
    return microstates

    with open('world_panel/microstates_list.txt', 'w') as f:
        f.write(str(microstates))
    print(f"Saved {len(microstates)} microstates to world_panel/microstates_list.txt")

if __name__ == "__main__":
    ms = get_microstates()
