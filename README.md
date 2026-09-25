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

## Docker

Build:

docker build -t churn-api .

Run:

docker run -p 8000:8000 churn-api