from pathlib import Path

import pandas as pd
import requests

DATA_FILE = Path("data") / "E Commerce Dataset.xlsx"

# Use local API for testing first
API_URL = "https://sea-lion-app-3sn49.ondigitalocean.app/batch-predict"

BATCH_SIZE = 100

# Load customer data
df = pd.read_excel(DATA_FILE, sheet_name="E Comm")

# Churn is the known result from the historical dataset,
# so it should not be sent to the prediction model.
df = df.drop(columns=["Churn"])

# Dataset used for this project does not retain CustomerID,
# so create an ID for tracking prediction results.
df.insert(0, "CustomerID", range(1, len(df) + 1))

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