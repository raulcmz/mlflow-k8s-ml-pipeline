import os

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException

from serving.schemas import PredictionRequest

app = FastAPI()

model = None


def get_required_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Required environment variable '{var_name}' is not set")
    return value


@app.on_event("startup")
def load_model():
    global model

    model_uri = get_required_env("MODEL_URI")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    print(f"Loading model from MODEL_URI={model_uri}")
    if tracking_uri:
        print(f"Using MLFLOW_TRACKING_URI={tracking_uri}")

    model = mlflow.sklearn.load_model(model_uri)
    print("Model loaded successfully!")


@app.get("/")
def root():
    return {"message": "MLflow remote inference API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: PredictionRequest):
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
    }