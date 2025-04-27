# Install necessary libraries if not already installed

# Imports
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

# 🛒 Load RetailData.csv (upload it to Colab if needed)
# Load RetailData.csv
df = pd.read_csv('RetailData.csv')


# 🏷️ Define X and y
# Assume 'Revenue' is the target
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 📋 Get all regression models
all_regressors = all_estimators(type_filter='regressor')

print(f"Testing {len(all_regressors)} models...\n")

# ⏱️ Start timer
start_time = time.time()

# Train and test each model
for name, RegressorClass in all_regressors:
    try:
        model = RegressorClass()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"{name}: R² = {r2:.4f}")
    except Exception as e:
        print(f"{name}: Failed ({e})")

# ⏱️ End timer
end_time = time.time()
print(f"\nTotal Runtime: {end_time - start_time:.2f} seconds")
