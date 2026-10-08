import joblib
import pandas as pd
import os
import psycopg2

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Customer Churn Prediction API")

# Load saved preprocessing + model pipeline
pipeline = joblib.load("models/churn_pipeline.joblib")
DATABASE_URL = os.getenv("DATABASE_URL")
def init_db():
    if not DATABASE_URL:
        return

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        id SERIAL PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        prediction INTEGER NOT NULL,
        churn_probability FLOAT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

    conn.commit()
    cur.close()
    conn.close()
init_db()

def save_prediction(customer_id, prediction, probability):
    if not DATABASE_URL:
        return

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO prediction_history (customer_id, prediction, churn_probability)
        VALUES (%s, %s, %s)
        """,
        (int(customer_id), int(prediction), float(probability))
    )

    conn.commit()
    cur.close()
    conn.close()
class CustomerData(BaseModel):
    Tenure: Optional[float] = None
    WarehouseToHome: Optional[float] = None
    HourSpendOnApp: Optional[float] = None
    PreferedOrderCat: str
    SatisfactionScore: int
    NumberOfAddress: int
    Complain: int
    OrderAmountHikeFromlastYear: Optional[float] = None
    CouponUsed: Optional[float] = None
    OrderCount: Optional[float] = None
    DaySinceLastOrder: Optional[float] = None
    CashbackAmount: float
    CustomerID: int


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerData):

    customer_data = customer.model_dump()
    customer_id = customer_data.pop("CustomerID")

    # Standardize category name
    if customer_data["PreferedOrderCat"] == "Mobile Phone":
        customer_data["PreferedOrderCat"] = "Mobile"

    customer_data = {
        key: (float("nan") if value is None else value)
        for key, value in customer_data.items()
    }

    customer_df = pd.DataFrame([customer_data])

    prediction = int(pipeline.predict(customer_df)[0])
    probability = float(
        pipeline.predict_proba(customer_df)[0][1]
    )
    save_prediction(customer_id, prediction, probability)

    return {
        "prediction": prediction,
        "churn_probability": probability,
        "risk": "At risk of churn"
        if prediction == 1
        else "Likely to remain active"
    }

@app.post("/batch-predict")
def batch_predict(customers: List[CustomerData]):
    results = []

    for customer in customers:
        customer_data = customer.model_dump()
        customer_id = customer_data.pop("CustomerID")

        if customer_data["PreferedOrderCat"] == "Mobile Phone":
            customer_data["PreferedOrderCat"] = "Mobile"

        customer_data = {
            key: (float("nan") if value is None else value)
            for key, value in customer_data.items()
        }

        customer_df = pd.DataFrame([customer_data])

        prediction = int(pipeline.predict(customer_df)[0])
        probability = float(
            pipeline.predict_proba(customer_df)[0][1]
        )

        save_prediction(customer_id, prediction, probability)

        results.append({
            "customer_id": customer_id,
            "prediction": prediction,
            "churn_probability": probability
        })

    return {
        "customers_processed": len(results),
        "predictions": results
    }