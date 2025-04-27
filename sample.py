import pandas as pd

# Sample data in CSV format

df = pd.DataFrame(data)

# Convert the datetime column to datetime objects
df['InvoiceDate'] = pd.to_datetime(df['datetime'], format='%m/%d/%y %H:%M')

# Create a column for the date (in m/dd/yy format)
df['date'] = df['datetime'].dt.strftime('%m/%d/%y')

# Create a column for the time in 12-hour format
df['time'] = df['datetime'].dt.strftime('%I:%M %p')

# Display the result
print(df)
