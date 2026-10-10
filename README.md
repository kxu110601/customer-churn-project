# E-Commerce Customer Churn Prediction

This project implements a machine learning service for predicting
customer churn in an e-commerce environment.

## Model

The system uses a Random Forest classifier trained on the
Ecommerce Customer Churn Analysis and Prediction dataset.

## Features

- Customer data preprocessing
- Missing-value imputation
- Categorical feature encoding
- Random Forest churn classification
- Saved preprocessing/model pipeline
- FastAPI inference endpoint
- Docker containerization

## API

Run the application locally with:

uvicorn app:app --reload

Interactive API documentation is available at:

http://127.0.0.1:8000/docs

## Deployed API

The Customer Churn Prediction API is deployed using DigitalOcean App Platform.

Interactive API documentation:

https://sea-lion-app-3sn49.ondigitalocean.app/docs

The API provides two prediction endpoints:

- `POST /predict` - generates a prediction for one customer.
- `POST /batch-predict` - generates predictions for multiple customers.

A prediction of `0` indicates that the customer is likely to remain active, while
a prediction of `1` indicates that the customer is at risk of churn.

## Running Batch Predictions

The repository includes the original e-commerce customer dataset and a `batch_predict.py` script for processing the complete dataset through the deployed API.

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Then run the batch prediction script:

```bash
python batch_predict.py
```

The script reads the customer data from:

```text
data/E Commerce Dataset.xlsx
```

The script submits all 5,630 customer records to the deployed `/batch-predict` endpoint. Progress is displayed while the script runs:

```text
Customers to process: 5630
Processed 100/5630 customers
Processed 200/5630 customers
...
Processed 5630/5630 customers
Batch prediction complete.
```

Prediction results are stored in the PostgreSQL database.

During the completed project run:

- 4,713 customers were predicted as non-churn (`0`).
- 917 customers were predicted as churn (`1`).

Running the batch prediction script again will add additional prediction records to the database.

## Docker

Build:

docker build -t churn-api .

Run:

docker run -p 8000:8000 churn-api