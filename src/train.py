import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn

# Import your preprocessing pipeline
from src.data_processing import build_preprocessing_pipeline


# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("data/processed/customer_features.csv")

y = df["is_high_risk"]

X_raw = df.drop(columns=["CustomerId", "is_high_risk"])


# -------------------------
# Feature Lists (must match Task 3)
# -------------------------
numerical_features = [
    "total_transaction_amount",
    "avg_transaction_amount",
    "transaction_count",
    "std_transaction_amount",
    "max_transaction_amount",
    "min_transaction_amount",
    "total_transaction_value",
    "avg_transaction_value",
    "avg_transaction_hour",
    "avg_transaction_day",
    "avg_transaction_month",
]

categorical_features = [
    "channel_id",
    "product_category",
    "provider_id",
    "pricing_strategy",
]


# -------------------------
# Build Preprocessing Pipeline
# -------------------------
pipeline = build_preprocessing_pipeline(
    numerical_features,
    categorical_features
)

X = pipeline.fit_transform(X_raw)


# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# Helper function (metrics)
# -------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


# -------------------------
# 1. Logistic Regression
# -------------------------
with mlflow.start_run(run_name="LogisticRegression"):

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    metrics = evaluate_model(lr, X_test, y_test)

    mlflow.log_params({"model": "LogisticRegression"})
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(lr, "model")

    print("Logistic Regression:", metrics)


# -------------------------
# 2. Random Forest
# -------------------------
with mlflow.start_run(run_name="RandomForest"):

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    rf.fit(X_train, y_train)

    metrics = evaluate_model(rf, X_test, y_test)

    mlflow.log_params({
        "model": "RandomForest",
        "n_estimators": 200,
        "max_depth": 10
    })

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(rf, "model")

    print("Random Forest:", metrics)


# -------------------------
# 3. Hyperparameter Tuning
# -------------------------
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, 20]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Params:", grid.best_params_)



# -------------------------
# 4. Best Model Logging
# -------------------------
with mlflow.start_run(run_name="BestModel"):

    metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    mlflow.log_params(grid.best_params_)
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="credit_risk_model"
    )

    print("Best Model:", metrics)
