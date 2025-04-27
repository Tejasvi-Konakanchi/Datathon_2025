import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker('en_GB')

# Base data from the sample
categories = ['Home & Kitchen', 'Beauty & Personal Care', 'Toys & Games', 
              'Electronics', 'Fashion', 'Sports & Outdoors', 'Food & Beverage', 'Automotive']
countries = ['United Kingdom', 'France', 'Germany', 'Spain', 'Italy', 'Netherlands']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']
times = ['8 AM', '9 AM', '10 AM', '11 AM', '12 PM', '1 PM', '2 PM', 
         '3 PM', '4 PM', '5 PM', '6 PM', '7 PM', '8 PM']

# Product database
products = [
    # Home & Kitchen
    {'StockCode': '22114', 'Description': 'HOT WATER BOTTLE TEA AND SYMPATHY', 'UnitPrice': 3.95, 'Category': 'Home & Kitchen'},
    {'StockCode': '21977', 'Description': 'PACK OF 60 PINK PAISLEY CAKE CASES', 'UnitPrice': 0.55, 'Category': 'Home & Kitchen'},
    {'StockCode': '21212', 'Description': 'PACK OF 72 RETROSPOT CAKE CASES', 'UnitPrice': 0.55, 'Category': 'Home & Kitchen'},
    {'StockCode': '21314', 'Description': 'SMALL GLASS HEART TRINKET POT', 'UnitPrice': 2.1, 'Category': 'Home & Kitchen'},
    {'StockCode': '84406B', 'Description': 'CREAM CUPID HEARTS COAT HANGER', 'UnitPrice': 2.75, 'Category': 'Home & Kitchen'},
    {'StockCode': '22803', 'Description': 'IVORY EMBROIDERED QUILT', 'UnitPrice': 35.75, 'Category': 'Home & Kitchen'},
    {'StockCode': '84029G', 'Description': 'KNITTED UNION FLAG HOT WATER BOTTLE', 'UnitPrice': 3.39, 'Category': 'Home & Kitchen'},
    {'StockCode': '37370', 'Description': 'RETRO COFFEE MUGS ASSORTED', 'UnitPrice': 1.06, 'Category': 'Home & Kitchen'},
    
    # Beauty & Personal Care
    {'StockCode': '15056BL', 'Description': 'EDWARDIAN PARASOL BLACK', 'UnitPrice': 4.95, 'Category': 'Beauty & Personal Care'},
    {'StockCode': '20679', 'Description': 'EDWARDIAN PARASOL RED', 'UnitPrice': 4.95, 'Category': 'Beauty & Personal Care'},
    {'StockCode': '19576', 'Description': 'LUXURY BATHROBE', 'UnitPrice': 15.99, 'Category': 'Beauty & Personal Care'},
    
    # Toys & Games
    {'StockCode': '22752', 'Description': 'SET 7 BABUSHKA NESTING BOXES', 'UnitPrice': 7.65, 'Category': 'Toys & Games'},
    {'StockCode': '21175', 'Description': 'GLOW IN THE DARK STICKERS', 'UnitPrice': 1.5, 'Category': 'Toys & Games'},
    {'StockCode': '12734', 'Description': 'EDUCATIONAL LEGO SET', 'UnitPrice': 9.99, 'Category': 'Toys & Games'},
    
    # Electronics
    {'StockCode': '19764', 'Description': 'PLAYSTATION 4 CONSOLE', 'UnitPrice': 299.99, 'Category': 'Electronics'},
    {'StockCode': '11235', 'Description': 'XBOX ONE CONTROLLER', 'UnitPrice': 49.99, 'Category': 'Electronics'},
    {'StockCode': '12058', 'Description': 'LOGITECH WIRELESS MOUSE', 'UnitPrice': 25.99, 'Category': 'Electronics'},
    {'StockCode': '11576', 'Description': 'COMPUTER SPEAKERS', 'UnitPrice': 45.0, 'Category': 'Electronics'},
    
    # Fashion
    {'StockCode': '22035', 'Description': 'PURSE WITH FLOWER DESIGN', 'UnitPrice': 12.0, 'Category': 'Fashion'},
    {'StockCode': '17854', 'Description': 'CASUAL DRESS', 'UnitPrice': 35.0, 'Category': 'Fashion'},
    {'StockCode': '21782', 'Description': 'STRIPED BEACH TOWEL', 'UnitPrice': 8.0, 'Category': 'Fashion'},
    
    # Sports & Outdoors
    {'StockCode': '25134', 'Description': 'OUTDOOR PATIO CHAIR SET', 'UnitPrice': 110.0, 'Category': 'Sports & Outdoors'},
    {'StockCode': '15943', 'Description': 'FITNESS TRACKER', 'UnitPrice': 50.0, 'Category': 'Sports & Outdoors'},
    
    # Food & Beverage
    {'StockCode': '32667', 'Description': 'REVOLUTIONARY SPORTS DRINK', 'UnitPrice': 2.5, 'Category': 'Food & Beverage'},
    
    # Automotive
    {'StockCode': '34452', 'Description': 'MULTIFUNCTIONAL TOOL SET', 'UnitPrice': 30.0, 'Category': 'Automotive'},
    {'StockCode': '23788', 'Description': 'UNIVERSAL CAR CHARGER', 'UnitPrice': 12.99, 'Category': 'Automotive'},
    {'StockCode': '11354', 'Description': 'STEEL WIRE BRUSH', 'UnitPrice': 8.0, 'Category': 'Automotive'},
]

# Generate more products
for i in range(100):
    category = random.choice(categories)
    if category == 'Home & Kitchen':
        desc = random.choice(['COFFEE MUG', 'CANDLE HOLDER', 'PILLOW COVER', 'KITCHEN UTENSIL', 'BATH TOWEL']) + ' ' + fake.color_name().upper()
        price = round(random.uniform(1.5, 40.0), 2)
    elif category == 'Beauty & Personal Care':
        desc = random.choice(['PERFUME', 'HAIR BRUSH', 'NAIL POLISH', 'FACE CREAM', 'SHAVING KIT']) + ' ' + fake.color_name().upper()
        price = round(random.uniform(5.0, 50.0), 2)
    elif category == 'Toys & Games':
        desc = random.choice(['PUZZLE', 'BOARD GAME', 'DOLL', 'ACTION FIGURE', 'CARD GAME']) + ' ' + fake.word().upper()
        price = round(random.uniform(3.0, 30.0), 2)
    elif category == 'Electronics':
        desc = random.choice(['SMARTPHONE', 'HEADPHONES', 'CHARGER', 'USB CABLE', 'SCREEN PROTECTOR']) + ' ' + fake.word().upper()
        price = round(random.uniform(10.0, 300.0), 2)
    elif category == 'Fashion':
        desc = random.choice(['T-SHIRT', 'JEANS', 'DRESS', 'JACKET', 'SCARF']) + ' ' + fake.color_name().upper()
        price = round(random.uniform(10.0, 100.0), 2)
    elif category == 'Sports & Outdoors':
        desc = random.choice(['YOGA MAT', 'DUMBBELL', 'RUNNING SHOES', 'BACKPACK', 'WATER BOTTLE']) + ' ' + fake.word().upper()
        price = round(random.uniform(15.0, 150.0), 2)
    elif category == 'Food & Beverage':
        desc = random.choice(['CHOCOLATE BAR', 'TEA BAGS', 'COFFEE BEANS', 'ENERGY DRINK', 'PROTEIN BAR']) + ' ' + fake.word().upper()
        price = round(random.uniform(1.0, 10.0), 2)
    else:  # Automotive
        desc = random.choice(['CAR WAX', 'AIR FRESHENER', 'TOW ROPE', 'JUMP STARTER', 'CLEANING SPRAY']) + ' ' + fake.word().upper()
        price = round(random.uniform(5.0, 50.0), 2)
    
    products.append({
        'StockCode': str(10000 + i),
        'Description': desc,
        'UnitPrice': price,
        'Category': category
    })

# Generate transactions
def generate_transactions(num_rows):
    data = []
    invoice_no = 536405  # Starting from the next invoice number
    
    for _ in range(num_rows):
        # Decide how many items in this invoice (1-8 items)
        items_in_invoice = random.randint(1, 8)
        customer_id = random.randint(13770, 20000)
        country = random.choice(countries)
        
        # Random date between 2010 and 2011
        year = 2010 if random.random() < 0.7 else 2011
        month = random.choice(months)
        weekday = random.choice(weekdays)
        time = random.choice(times)
        day_type = 'Weekday' if weekday in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] else 'Weekend'
        
        for _ in range(items_in_invoice):
            product = random.choice(products)
            quantity = random.randint(1, 12)
            revenue = round(quantity * product['UnitPrice'], 2)
            
            data.append({
                'InvoiceNo': invoice_no,
                'StockCode': product['StockCode'],
                'Description': product['Description'],
                'Quantity': quantity,
                'UnitPrice': product['UnitPrice'],
                'CustomerID': customer_id,
                'Country': country,
                'Revenue': revenue,
                'Year': year,
                'Month': month,
                'DayOfWeek': weekday,
                'Hour': time,
                'Weekday': day_type,
                'Category': product['Category']
            })
        
        invoice_no += 1
    
    return data

# Generate 5000 rows
new_data = generate_transactions(100000)

# Convert to DataFrame
df = pd.DataFrame(new_data)

# Save to CSV
df.to_csv('generated_retail_data.csv', index=False)

print(f"Generated {len(df)} rows of retail data.")