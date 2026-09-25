import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

df = pd.read_excel(
    "data/E Commerce Dataset.xlsx",
    sheet_name="E Comm"
)

# Clean inconsistent category names
df["PreferedOrderCat"] = df["PreferedOrderCat"].replace(
    "Mobile Phone",
    "Mobile"
)

# Separate features from target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Convert categorical column into numeric columns
X = pd.get_dummies(
    X,
    columns=["PreferedOrderCat"],
    drop_first=False,
    dtype=int
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Numeric columns with missing values
missing_columns = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder"
]

# Fill missing values using the median from the training set
imputer = SimpleImputer(strategy="median")

X_train[missing_columns] = imputer.fit_transform(
    X_train[missing_columns]
)

X_test[missing_columns] = imputer.transform(
    X_test[missing_columns]
)

print("Missing values in training data:")
print(X_train.isnull().sum())

print()
print("Missing values in test data:")
print(X_test.isnull().sum())

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate model
print()
print("Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print()
print("Classification Report:")
print(classification_report(y_test, y_pred))