from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

DATA_FILE = Path("data") / "E Commerce Dataset.xlsx"

# Use deployed API for testing first
API_URL = "https://sea-lion-app-3sn49.ondigitalocean.app/batch-predict"

BATCH_SIZE = 100

# Load customer data
df = pd.read_excel(DATA_FILE, sheet_name="E Comm")

# Standardize category name the same way as model training
df["PreferedOrderCat"] = df["PreferedOrderCat"].replace(
    "Mobile Phone",
    "Mobile"
)

# Separate features and actual churn result
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Recreate the exact same 70/30 split used during model training
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Only send the held-out test customers to the deployed model
df = X_test.copy()

# Create IDs using their original dataset row numbers
df.insert(0, "CustomerID", df.index + 1)

# Convert pandas NaN values to Python None.
# requests will encode None as JSON null.
df = df.astype(object).where(pd.notnull(df), None)

records = df.to_dict(orient="records")

print(f"Customers to process: {len(records)}")

total_processed = 0

for start in range(0, len(records), BATCH_SIZE):
    batch = records[start:start + BATCH_SIZE]

    response = requests.post(
        API_URL,
        json=batch,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()
    processed = result["customers_processed"]
    total_processed += processed

    print(
        f"Processed {total_processed}/{len(records)} customers"
    )

print("Batch prediction complete.")