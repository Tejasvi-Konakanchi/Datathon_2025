import pandas as pd
import os
import re

# Load your dataset
df = pd.read_csv("retail_cleaned_data.csv", dtype={"InvoiceNo": "string", "StockCode": "string","Description": "string", "Quantity": int, "UnitPrice": float, "Country": "string", "Revenue": float, "Year": "string", "Month": "string", "DayOfWeek": "string", "Hour": "string", "Year": "string", "Weekday": "string"})

df['Hour'] = df['Hour'].astype(int).apply(
    lambda hour: (
        "12 AM" if hour == 0 else
        f"{hour} AM" if hour < 12 else
        "12 PM" if hour == 12 else
        f"{hour - 12} PM"
    )
)

df['DayOfWeek'] = df['DayOfWeek'].astype(int).apply(
    lambda day: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day]
)

df.to_csv('CleanedData.csv', index=False)

cdf = pd.read_csv("CleanedData.csv", dtype={"InvoiceNo": "string", "StockCode": "string","Description": "string", "Quantity": int, "UnitPrice": float, "Country": "string", "Revenue": float, "Year": "string", "Month": "string", "DayOfWeek": "string", "Hour": "string", "Year": "string", "Weekday": "string"})

# Create folders if they don't exist
os.makedirs('products', exist_ok=True)
os.makedirs('countries', exist_ok=True)

# Select certain columns to
selected_columns = [
    'Description', 'Quantity', 'UnitPrice', 
    'Revenue', 'Year', 'Month', 
    'DayOfWeek', 'Hour', 'Weekday'
]

# Split by description

for description, product_df in cdf.groupby('Description'):
    safe_description = re.sub(r'[\\/*?:"<>|]', "_", description)
    filename = f'products/{safe_description}.csv'
    product_df_selected = product_df[selected_columns]
    product_df_selected.to_csv(filename, index=False)

    

for description, country_df in cdf.groupby('Country'):
    safe_description = re.sub(r'[\\/*?:"<>|]', "_", description)
    filename = f'countries/{safe_description}.csv'
    country_df_selected = country_df[selected_columns]
    country_df_selected.to_csv(filename, index=False)