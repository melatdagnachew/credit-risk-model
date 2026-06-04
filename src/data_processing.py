import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(file_path):
    """
    Load raw transaction data.
    """
    return pd.read_csv(file_path)


def create_aggregate_features(df):
    """
    Create customer-level aggregate features.
    """

    agg_df = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("TransactionId", "count"),
            std_transaction_amount=("Amount", "std"),
            max_transaction_amount=("Amount", "max"),
            min_transaction_amount=("Amount", "min"),
            total_transaction_value=("Value", "sum"),
            avg_transaction_value=("Value", "mean"),
        )
        .reset_index()
    )

    return agg_df


def extract_time_features(df):
    """
    Extract time-based features from TransactionStartTime.
    """

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year

    return df


def create_customer_time_features(df):
    """
    Aggregate time features per customer.
    """

    time_df = (
        df.groupby("CustomerId")
        .agg(
            avg_transaction_hour=("transaction_hour", "mean"),
            avg_transaction_day=("transaction_day", "mean"),
            avg_transaction_month=("transaction_month", "mean"),
        )
        .reset_index()
    )

    return time_df


def create_customer_categorical_features(df):
    """
    Keep most frequent category per customer.
    """

    cat_df = (
        df.groupby("CustomerId")
        .agg(
            channel_id=("ChannelId", lambda x: x.mode()[0]),
            product_category=("ProductCategory", lambda x: x.mode()[0]),
            provider_id=("ProviderId", lambda x: x.mode()[0]),
            pricing_strategy=("PricingStrategy", lambda x: x.mode()[0]),
        )
        .reset_index()
    )

    return cat_df


def build_customer_dataset(df):
    """
    Build customer-level dataset.
    """

    df = extract_time_features(df)

    agg_df = create_aggregate_features(df)

    time_df = create_customer_time_features(df)

    cat_df = create_customer_categorical_features(df)

    customer_df = (
        agg_df
        .merge(time_df, on="CustomerId")
        .merge(cat_df, on="CustomerId")
    )

    return customer_df


def build_preprocessing_pipeline(
    numerical_features,
    categorical_features
):
    """
    Build sklearn preprocessing pipeline.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor)
        ]
    )


if __name__ == "__main__":

    df = load_data("data/raw/data.csv")

    customer_df = build_customer_dataset(df)

    customer_df.to_csv(
    "data/processed/customer_features.csv",
    index=False
    )

    print("Customer features saved successfully")

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

    pipeline = build_preprocessing_pipeline(
        numerical_features,
        categorical_features,
    )

    transformed = pipeline.fit_transform(customer_df)

    print("Pipeline fitted successfully")
    print(transformed.shape)
