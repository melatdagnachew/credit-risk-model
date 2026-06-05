import mlflow
import pandas as pd

from fastapi import FastAPI

from src.api.pydantic_models import (
    CustomerData,
    PredictionResponse,
)

app = FastAPI(
    title="Credit Risk API"
)

MODEL_NAME = "credit_risk_model"

model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/latest"
)


@app.get("/")
def health_check():
    return {
        "status": "running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: CustomerData):

    df = pd.DataFrame(
        [data.dict()]
    )

    probability = float(
        model.predict(df)[0]
    )

    prediction = (
        1 if probability >= 0.5 else 0
    )

    return PredictionResponse(
        risk_probability=probability,
        prediction=prediction
    )
