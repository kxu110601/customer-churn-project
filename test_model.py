import pandas as pd
import joblib

# Load saved preprocessing + model pipeline
pipeline = joblib.load("models/churn_pipeline.joblib")

# Example new customer
customer = pd.DataFrame([{
    "Tenure": 5,
    "WarehouseToHome": 15,
    "HourSpendOnApp": 3,
    "PreferedOrderCat": "Mobile",
    "SatisfactionScore": 2,
    "NumberOfAddress": 3,
    "Complain": 1,
    "OrderAmountHikeFromlastYear": 12,
    "CouponUsed": 1,
    "OrderCount": 2,
    "DaySinceLastOrder": 25,
    "CashbackAmount": 120
}])

# Make prediction
prediction = pipeline.predict(customer)[0]
probability = pipeline.predict_proba(customer)[0][1]

print("Prediction:", prediction)
print("Churn probability:", probability)

if prediction == 1:
    print("Customer is predicted to be at risk of churn.")
else:
    print("Customer is predicted to remain active.")