import pandas as pd
from sklearn.linear_model import LinearRegression

# Read the data from CSV (replace this with your actual CSV path)
df = pd.read_csv('CleanedData.csv')

# Dictionary to map month names to numbers
month_dict = {
    'January': '01', 'February': '02', 'March': '03', 'April': '04', 'May': '05', 'June': '06',
    'July': '07', 'August': '08', 'September': '09', 'October': '10', 'November': '11', 'December': '12'
}

# Ensure 'Year' and 'Month' are in string format (in case they are not)
df['Year'] = df['Year'].astype(str)
df['Month'] = df['Month'].astype(str)

# Convert month names to numbers
df['Month'] = df['Month'].map(month_dict)

# Combine 'Year' and 'Month' into a single 'month_year' column in 'YYYY-MM' format
df['month_year'] = df['Year'] + '-' + df['Month']

# Convert 'month_year' to datetime format
df['month_year'] = pd.to_datetime(df['month_year'], format='%Y-%m')

# Check the result by printing the first few rows
print(df[['Year', 'Month', 'month_year']].head())

# --- Popularity Score Calculations ---
# Step 1: Calculate total revenue for each product
product_revenue = df.groupby('Description')['Revenue'].sum().reset_index()

# Step 2: Calculate the number of unique customers for each product
unique_customers = df.groupby('Description')['CustomerID'].nunique().reset_index()
unique_customers = unique_customers.rename(columns={'CustomerID': 'UniqueCustomers'})

# Step 3: Calculate the slope (trend) for each product using linear regression
def calculate_slope(product_df):
    # Create a numerical representation of time (e.g., number of months)
    product_df['MonthIndex'] = pd.to_datetime(product_df['month_year']).apply(
        lambda x: x.month + 12 * (x.year - product_df['month_year'].min().year)
    )
    
    # Perform a linear regression (Revenue as the dependent variable, MonthIndex as independent)
    X = product_df['MonthIndex'].values.reshape(-1, 1)
    y = product_df['Revenue'].values
    model = LinearRegression()
    model.fit(X, y)
    
    return model.coef_[0]  # Return the slope
print(product_revenue.columns)
print(unique_customers.columns)

# Apply the slope calculation to each product
product_slope = df.groupby('Description').apply(calculate_slope).reset_index(name='Slope')

# Step 4: Merge all the information into one DataFrame
product_data = pd.merge(product_revenue, unique_customers, on='Description')
product_data = pd.merge(product_data, product_slope, on='Description')

# Step 5: Calculate the popularity score using the formula
product_data['popularity_score'] = product_data['Revenue'] * (1.1 ** product_data['UniqueCustomers']) * product_data['Slope']

# Step 6: Print the results
print(product_data[['Description', 'popularity_score']])

# Optionally, save the final DataFrame to a new CSV
product_data.to_csv('Product_Popularity_Scores.csv', index=False)

# --- Additional Analysis ---
# Grouping by month_year and calculating total revenue for each month
monthly_revenue = df.groupby('month_year')['Revenue'].sum().reset_index()

# Print the grouped results
print(monthly_revenue.head())
