import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from sklearn.metrics import accuracy_score  # Import accuracy_score from sklearn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Example of loading your dataset (assuming you have it as a CSV)
df = pd.read_csv("generated_retail_data.csv")

# Define the categories
category_labels = [
    "Electronics", "Fashion", "Home & Kitchen", "Sports & Outdoors", "Beauty & Personal Care",
    "Health", "Toys & Games", "Books & Stationery", "Automotive", "Food & Beverage",
    "Baby & Kids", "Pet Supplies", "Books", "Arts & Crafts", "Technology & Gadgets"
]

# Create a mapping from category to integer
label_map = {category: idx for idx, category in enumerate(category_labels)}

# Map the 'Category' column to integer labels
df['Category'] = df['Category'].map(label_map)

# Check if there are any NaN values after mapping
if df['Category'].isnull().any():
    print("Warning: There are some categories that could not be mapped.")

# Split dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['Description'], padding="max_length", truncation=True, max_length=128)

# Convert to Dataset objects
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.texts = dataframe['Description'].values
        self.labels = dataframe['Category'].values  # Labels are already mapped to integers

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        text = item['Description']
        label = torch.tensor(item['Category'], dtype=torch.long)  # Ensure 'Category' is an integer
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128)
        encoding['labels'] = label
        return encoding

# Prepare the datasets
train_dataset = CustomDataset(train_df)
val_dataset = CustomDataset(val_df)

# Define model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(category_labels))

# Define the compute_metrics function to calculate accuracy using sklearn
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)  # Get the class with the highest probability
    return {'accuracy': accuracy_score(labels, predictions)}

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="steps",  # Evaluate at specific steps
    eval_steps=500,  # Every 500 steps (you can change this depending on your dataset size)
    load_best_model_at_end=True,  # Load the best model at the end of training
)

# Initialize Trainer with compute_metrics to track accuracy
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # Add the metric function here
)

# Train the model
trainer.train()

# Save the model and tokenizer after training
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

print("Model and tokenizer have been saved successfully.")
