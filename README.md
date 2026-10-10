# E-Commerce Customer Churn Prediction

This project implements a machine learning service for predicting customer churn in an e-commerce environment.

The system uses a Random Forest classifier trained on the E-Commerce Customer Churn Analysis and Prediction dataset. The trained model is exposed through a FastAPI REST API and deployed using DigitalOcean App Platform.

## Model

The Random Forest model uses customer information such as tenure, order activity, satisfaction, complaints, coupons, and cashback amount to predict whether a customer is likely to churn.

The complete dataset contains 5,630 customer records.

The dataset is divided into:

- 70% training data
- 30% testing data

The held-out testing set contains 1,689 customer records and is not used to train the model.

A prediction of:

- `0` means the customer is predicted to remain active.
- `1` means the customer is predicted to churn.

## Features

The project includes:

- Customer data preprocessing
- Missing-value imputation
- Categorical feature encoding
- Random Forest churn classification
- Saved preprocessing/model pipeline
- FastAPI REST API
- Single-customer prediction
- Batch prediction
- PostgreSQL prediction history
- Docker containerization
- DigitalOcean App Platform deployment

## Deployed API

The Customer Churn Prediction API is deployed using DigitalOcean App Platform.

Interactive API documentation is available at:

https://sea-lion-app-3sn49.ondigitalocean.app/docs

The deployed API can be tested directly from a web browser without running the project locally.

The API provides two prediction endpoints:

- `POST /predict` - generates a prediction for one customer.
- `POST /batch-predict` - generates predictions for multiple customers.

### Testing the Deployed API

1. Open:

   https://sea-lion-app-3sn49.ondigitalocean.app/docs

2. Expand `POST /predict`.

3. Click **Try it out**.

4. Replace the request body with one of the example customers below.

5. Click **Execute**.

6. A successful request will return HTTP status `200` along with the predicted churn class, churn probability, and risk classification.

A prediction of `0` means that the customer is predicted to remain active.  
A prediction of `1` means that the customer is predicted to churn.

#### Example Customer 1

```json
{
  "Tenure": 12,
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
  "CashbackAmount": 120,
  "CustomerID": 1001
}
```

#### Example Customer 2

```json
{
  "Tenure": 20,
  "WarehouseToHome": 8,
  "HourSpendOnApp": 4,
  "PreferedOrderCat": "Laptop & Accessory",
  "SatisfactionScore": 4,
  "NumberOfAddress": 2,
  "Complain": 0,
  "OrderAmountHikeFromlastYear": 10,
  "CouponUsed": 2,
  "OrderCount": 8,
  "DaySinceLastOrder": 3,
  "CashbackAmount": 180,
  "CustomerID": 1002
}
```

#### Example Customer 3

This example demonstrates that the API can also process supported missing values.

```json
{
  "Tenure": null,
  "WarehouseToHome": 15,
  "HourSpendOnApp": 3,
  "PreferedOrderCat": "Mobile",
  "SatisfactionScore": 2,
  "NumberOfAddress": 3,
  "Complain": 1,
  "OrderAmountHikeFromlastYear": 12,
  "CouponUsed": null,
  "OrderCount": 2,
  "DaySinceLastOrder": 25,
  "CashbackAmount": 120,
  "CustomerID": 1003
}
```

The preprocessing pipeline handles supported missing values before the customer record is passed to the Random Forest model.

An example response is:

```json
{
  "prediction": 0,
  "churn_probability": 0.28,
  "risk": "Likely to remain active"
}
```

The exact prediction and churn probability depend on the customer data supplied to the API.

### Testing Batch Predictions Through the API

The `POST /batch-predict` endpoint accepts multiple customer records in a single request.

1. Expand `POST /batch-predict`.

2. Click **Try it out**.

3. Replace the request body with the following example:

```json
[
  {
    "Tenure": 12,
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
    "CashbackAmount": 120,
    "CustomerID": 1001
  },
  {
    "Tenure": 20,
    "WarehouseToHome": 8,
    "HourSpendOnApp": 4,
    "PreferedOrderCat": "Laptop & Accessory",
    "SatisfactionScore": 4,
    "NumberOfAddress": 2,
    "Complain": 0,
    "OrderAmountHikeFromlastYear": 10,
    "CouponUsed": 2,
    "OrderCount": 8,
    "DaySinceLastOrder": 3,
    "CashbackAmount": 180,
    "CustomerID": 1002
  }
]
```

4. Click **Execute**.

The API will return a prediction for each customer in the request.

For testing the complete held-out testing set of 1,689 customers, use the `batch_predict.py` script described in the next section.

## Running Batch Predictions

The repository includes the original e-commerce customer dataset and a `batch_predict.py` script.

The script recreates the same 70/30 train/test split used during model training and sends only the held-out 30% testing set to the deployed API.

This results in 1,689 customer records being processed.

### Step 1 - Clone the Repository

Clone the repository and enter the project directory:

```bash
git clone https://github.com/kxu110601/customer-churn-project.git
cd customer-churn-project
```

### Step 2 - Install Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 3 - Run the Batch Prediction Script

Run:

```bash
python batch_predict.py
```

The script reads the customer data from:

```text
data/E Commerce Dataset.xlsx
```

The script then recreates the same held-out test set used during model evaluation and submits the records to the deployed `/batch-predict` endpoint in batches.

Expected output:

```text
Customers to process: 1689
Processed 100/1689 customers
Processed 200/1689 customers
...
Processed 1600/1689 customers
Processed 1689/1689 customers
Batch prediction complete.
```

During the completed project test run:

- 1,436 customers were predicted as non-churn (`0`).
- 253 customers were predicted as churn (`1`).
- 1,689 total test customers were processed.

Prediction results generated by the deployed API are stored in the PostgreSQL `prediction_history` table.

Running the batch prediction script again will add additional prediction records to the database.

## Running the API Locally

The API can also be run directly with Python.

### Step 1 - Install Dependencies

From the project directory:

```bash
pip install -r requirements.txt
```

### Step 2 - Start the API

Run:

```bash
uvicorn app:app --reload
```

The terminal should display a message similar to:

```text
Uvicorn running on http://127.0.0.1:8000
```

### Step 3 - Open the API Documentation

Open the following address in a browser:

```text
http://127.0.0.1:8000/docs
```

The FastAPI Swagger interface can then be used to test the `/predict` and `/batch-predict` endpoints.

Press `Ctrl+C` in the terminal to stop the API.

## Running the API with Docker

Docker can be used to run the API without manually configuring the Python runtime inside the container.

### Prerequisite

Install and start Docker Desktop.

Verify that Docker is running:

```bash
docker --version
```

### Step 1 - Clone the Repository

If the repository has not already been downloaded:

```bash
git clone https://github.com/kxu110601/customer-churn-project.git
cd customer-churn-project
```

If the repository is already downloaded, open a terminal in the `customer-churn-project` directory.

### Step 2 - Build the Docker Image

Run:

```bash
docker build -t customer-churn-api .
```

Docker will read the `Dockerfile`, install the required dependencies, copy the application and trained model into the image, and create an image named:

```text
customer-churn-api
```

To confirm that the image was created:

```bash
docker images
```

The `customer-churn-api` image should appear in the list.

### Step 3 - Start the Docker Container

Run:

```bash
docker run --rm -p 8000:8000 customer-churn-api
```

The options used are:

- `--rm` removes the container automatically after it is stopped.
- `-p 8000:8000` maps port 8000 on the local computer to port 8000 inside the container.
- `customer-churn-api` specifies the Docker image to run.

The terminal should show that Uvicorn has started.

### Step 4 - Open the Containerized API

While the Docker container is running, open:

```text
http://127.0.0.1:8000/docs
```

The FastAPI Swagger interface should appear.

### Step 5 - Test a Prediction

Expand:

```text
POST /predict
```

Click:

```text
Try it out
```

Enter customer information in the request body and click:

```text
Execute
```

A successful request returns an HTTP `200` response containing the prediction and churn probability.

The `/batch-predict` endpoint can also be tested by providing multiple customer records in the request body.

### Step 6 - Stop the Container

Return to the terminal running Docker and press:

```text
Ctrl+C
```

Because the container was started with `--rm`, Docker will remove the stopped container automatically.

### Running Docker in the Background

The container can alternatively be started in detached mode:

```bash
docker run -d --name customer-churn-api-container -p 8000:8000 customer-churn-api
```

Confirm that it is running:

```bash
docker ps
```

Open:

```text
http://127.0.0.1:8000/docs
```

To stop the detached container:

```bash
docker stop customer-churn-api-container
```

To remove it:

```bash
docker rm customer-churn-api-container
```

## PostgreSQL Database

The deployed application uses PostgreSQL to store prediction history.

Each stored prediction contains:

- Customer ID
- Predicted class
- Churn probability
- Timestamp

The application obtains the database connection string from the `DATABASE_URL` environment variable rather than storing database credentials directly in the source code.

When the API is run locally or through Docker without a `DATABASE_URL`, prediction functionality still works, but prediction history is not written to the deployed PostgreSQL database.

## Project Files

```text
customer-churn-project/
│
├── data/
│   └── E Commerce Dataset.xlsx
│
├── models/
│   └── churn_pipeline.joblib
│
├── app.py
├── batch_predict.py
├── preprocess.py
├── train_model.py
├── test_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

### Important Files

- `app.py` - FastAPI application and prediction endpoints.
- `batch_predict.py` - recreates the held-out 30% test set and sends it to the deployed API.
- `train_model.py` - trains the Random Forest model.
- `preprocess.py` - performs dataset preprocessing.
- `test_model.py` - contains model testing functionality.
- `models/churn_pipeline.joblib` - saved preprocessing and Random Forest pipeline.
- `data/E Commerce Dataset.xlsx` - project dataset.
- `Dockerfile` - defines the Docker container used to run the API.
- `requirements.txt` - Python dependencies required by the project.