from fastapi import FastAPI
import mlflow

import pandas as pd

from serving.schemas import PredictionRequest

app = FastAPI()

# 👉 IMPORTANTE: pon aquí la ruta correcta del modelo
MODEL_URI = "runs:/d73cdf49384e4bc8b3faccba9370e4d2/model"

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