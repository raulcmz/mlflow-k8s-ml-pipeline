from fastapi import FastAPI
import mlflow

import pandas as pd

from serving.schemas import PredictionRequest

app = FastAPI()

# 👉 IMPORTANTE: pon aquí la ruta correcta del modelo
MODEL_URI = "/home/raul/mlflow-k8s-ml-pipeline/mlruns/1/models/m-89cd323af97d43f9aba93bb1fc4feaa6/artifacts"

model = None


@app.on_event("startup")
def load_model():
    global model
    print("Loading model from MLflow...")
    model = mlflow.sklearn.load_model(MODEL_URI)
    print("Model loaded successfully!")


@app.get("/")
def root():
    return {"message": "MLflow MLOps API is running"}


@app.post("/predict")
def predict(data: PredictionRequest):
    df = pd.DataFrame([data.dict()])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
    }