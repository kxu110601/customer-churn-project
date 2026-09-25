import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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


# Load dataset
df = pd.read_excel(
    "data/E Commerce Dataset.xlsx",
    sheet_name="E Comm"
)

# Standardize inconsistent category name
df["PreferedOrderCat"] = df["PreferedOrderCat"].replace(
    "Mobile Phone",
    "Mobile"
)

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Identify categorical and numeric features
categorical_features = ["PreferedOrderCat"]

numeric_features = [
    column
    for column in X.columns
    if column not in categorical_features
]

# Numeric preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)

# Full preprocessing + model pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Train entire pipeline
pipeline.fit(X_train, y_train)

# Test predictions
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Create models directory
Path("models").mkdir(exist_ok=True)

# Save complete preprocessing + model pipeline
joblib.dump(
    pipeline,
    "models/churn_pipeline.joblib"
)

print()
print("Saved model to models/churn_pipeline.joblib")