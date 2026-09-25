import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Customer Churn Prediction API")

# Load saved preprocessing + model pipeline
pipeline = joblib.load("models/churn_pipeline.joblib")


class CustomerData(BaseModel):
    Tenure: float
    WarehouseToHome: float
    HourSpendOnApp: float
    PreferedOrderCat: str
    SatisfactionScore: int
    NumberOfAddress: int
    Complain: int
    OrderAmountHikeFromlastYear: float
    CouponUsed: float
    OrderCount: float
    DaySinceLastOrder: float
    CashbackAmount: float


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerData):

    customer_data = customer.model_dump()

    # Standardize category name
    if customer_data["PreferedOrderCat"] == "Mobile Phone":
        customer_data["PreferedOrderCat"] = "Mobile"

    customer_df = pd.DataFrame([customer_data])

    prediction = int(pipeline.predict(customer_df)[0])
    probability = float(
        pipeline.predict_proba(customer_df)[0][1]
    )

    return {
        "prediction": prediction,
        "churn_probability": probability,
        "risk": "At risk of churn"
        if prediction == 1
        else "Likely to remain active"
    }