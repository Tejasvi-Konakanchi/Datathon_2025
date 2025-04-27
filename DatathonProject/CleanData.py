import pandas as pd
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import tqdm
import pyarrow as pa

from datasets import Dataset

# Read the original CSV file
df = pd.read_csv('Mode_Craft_Ecommerce_Data - Online_Retail.csv')  # Replace 'input_file.csv' with your actual file name

# Convert the datetime column to datetime objects (adjust the format if necessary)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%y %H:%M')

# Create new columns for the date and time
df['date'] = df['InvoiceDate'].dt.strftime('%m/%d/20%y')
df['time'] = df['InvoiceDate'].dt.strftime('%I:%M %p')

# Save the updated DataFrame with the new columns to a new CSV file
df.to_csv('CleanedData.csv', index=False)

print("CSV file with new date and time columns saved successfully!")

# Load the training dataset to get the descriptions
cdf = pd.read_csv("CleanedData.csv", dtype={"InvoiceNo": "string", "StockCode": "string","Description": "string", "Quantity": int, "UnitPrice": float, "Country": "string", "Revenue": float, "Year": "string", "Month": "string", "DayOfWeek": "string", "Hour": "string", "Year": "string", "Weekday": "string"})

descriptions = cdf['Description'].fillna("").tolist()

# Load the trained model and tokenizer
model_path = './trained_model'  # Adjust this path to where your model is saved
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Ensure the model is on the correct device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# Prepare the dataset using Hugging Face datasets library
dataset = Dataset.from_dict({"description": descriptions})

# Define the candidate labels (these should match the categories you trained on)
candidate_labels = [
    "Electronics", "Fashion", "Home & Kitchen", "Sports & Outdoors", "Beauty & Personal Care",
    "Health", "Toys & Games", "Books & Stationery", "Automotive", "Food & Beverage",
    "Baby & Kids", "Pet Supplies", "Books", "Arts & Crafts", "Technology & Gadgets"
]

# Create a DataLoader for batch processing
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size)

# Function to classify each batch using the trained model
def classify_batch(batch):
    texts = batch['description']
    
    # Tokenize the batch
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class for each text (class with the highest logit score)
    logits = outputs.logits
    predicted_class_ids = torch.argmax(logits, dim=-1)
    
    # Convert tensor to a list of predictions
    return predicted_class_ids.cpu().numpy()

# List to hold the predicted categories
predicted_categories = []

# Process the dataset in batches
for batch in tqdm.tqdm(dataloader, desc="Processing batches", unit="batch"):
    predicted_categories.extend(classify_batch(batch))

# Map the predicted class indices to category labels
category_labels = [
    "Electronics", "Fashion", "Home & Kitchen", "Sports & Outdoors", "Beauty & Personal Care",
    "Health", "Toys & Games", "Books & Stationery", "Automotive", "Food & Beverage",
    "Baby & Kids", "Pet Supplies", "Books", "Arts & Crafts", "Technology & Gadgets"
]

# Map predicted indices to labels
predicted_category_labels = [category_labels[idx] for idx in predicted_categories]

# Add the predictions to the DataFrame
cdf['Predicted_Category'] = predicted_category_labels

# Save the predictions to a new CSV
cdf.to_csv('predicted_categories.csv', index=False)

# Create folders if they don't exist
os.makedirs('products', exist_ok=True)
os.makedirs('countries', exist_ok=True)

# Select certain columns to save
selected_columns = ['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country','date','time','Predicted_Category']

# Save product-specific CSVs
for description, product_df in cdf.groupby('Description'):
    safe_description = re.sub(r'[\\/*?:"<>|]', "_", description)
    filename = f'products/{safe_description}.csv'
    product_df_selected = product_df[selected_columns]
    product_df_selected.to_csv(filename, index=False)

# Save country-specific CSVs
for country, country_df in cdf.groupby('Country'):
    safe_country = re.sub(r'[\\/*?:"<>|]', "_", country)
    filename = f'countries/{safe_country}.csv'
    country_df_selected = country_df[selected_columns]
    country_df_selected.to_csv(filename, index=False)

print("Predictions and CSV files have been saved.")