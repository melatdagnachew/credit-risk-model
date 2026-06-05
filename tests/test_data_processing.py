import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_processing import load_data, build_customer_dataset


def test_customer_aggregation():
    df = load_data("data/raw/data.csv")

    customer_df = build_customer_dataset(df)

    assert customer_df["CustomerId"].nunique() == len(customer_df)
