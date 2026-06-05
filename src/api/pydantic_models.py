from pydantic import BaseModel


class CustomerData(BaseModel):
    total_transaction_amount: float
    avg_transaction_amount: float
    transaction_count: int
    std_transaction_amount: float
    max_transaction_amount: float
    min_transaction_amount: float
    total_transaction_value: float
    avg_transaction_value: float
    avg_transaction_hour: float
    avg_transaction_day: float
    avg_transaction_month: float

    channel_id: str
    product_category: str
    provider_id: str
    pricing_strategy: float


class PredictionResponse(BaseModel):
    risk_probability: float
    prediction: int
