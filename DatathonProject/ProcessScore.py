import pandas as pd
from scipy.stats import linregress

# Load your CSV
df = pd.read_csv('CleanedData.csv')

# Assume there's a 'month_year' column like '2024-01', '2024-02', etc.
# If it's in another format, convert it to strings like 'YYYY-MM'

# Group by product and month_year
grouped = df.groupby(['Description', 'month_year'])

# Calculate total revenue per product per month (you may already have this)
monthly_revenue = grouped['revenue'].sum().reset_index()

# Create a dictionary to store slopes for each product
slopes = {}

# Now, calculate the slope for each product based on their monthly revenue
for product_id, group in monthly_revenue.groupby('Description'):
    # Convert 'month_year' to numerical values
    group['month_num'] = range(len(group))  # e.g., 0, 1, 2, ...

    # Perform linear regression to calculate the slope (month_num vs revenue)
    regression = linregress(group['month_num'], group['revenue'])
    slopes[product_id] = regression.slope

# Max absolute slope for normalization
max_abs_slope = max(abs(s) for s in slopes.values()) or 1  # Avoid division by 0

# Calculate final scores for each product
scores = []
alpha = 0.7
beta = 0.3

# Group by product to calculate the scores
for product_id, group in df.groupby('product_id'):
    total_revenue = group['revenue'].sum()
    unique_customers = group['customer_id'].nunique()

    slope = slopes.get(product_id, 0)  # Get slope for the product (default 0 if not found)

    # Normalize the slope (to be between -1 and 1)
    normalized_slope = slope / max_abs_slope

    # Calculate the score
    score = (total_revenue ** alpha) * (unique_customers ** beta) * (1 + normalized_slope)

    scores.append({'product_id': product_id, 'score': score})

# Create a DataFrame to hold the scores and save to CSV
scores_df = pd.DataFrame(scores)
scores_df.to_csv('product_scores.csv', index=False)
