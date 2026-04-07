from mlflow import MlflowClient
from training.validate import validate_raw_data
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

REGISTERED_MODEL_NAME = "telco-churn-model"

def load_data(path: str) -> pd.DataFrame:
    """
    Load the raw dataset from a CSV file.
    """
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Telco Churn dataset.
    """
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


def split_features_target(df: pd.DataFrame):
    """
    Split cleaned DataFrame into features (X) and target (y).
    """
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build a preprocessing pipeline for numeric and categorical columns.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor

def get_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }


def evaluate_model(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_and_log_model(
    model_name,
    model,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test,
):
    with mlflow.start_run(run_name=model_name) as run:

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = evaluate_model(y_test, y_pred, y_proba)

        # Log params
        mlflow.log_param("model_name", model_name)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model
        model_info = mlflow.sklearn.log_model(pipeline, name="model")

        print(f"\nModel: {model_name}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        return {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "metrics": metrics,
            "model_uri": model_info.model_uri,
        }
    
def register_best_model(best_result: dict):
    """
    Register the best model in MLflow Model Registry and assign the 'champion' alias.
    """
    client = MlflowClient()

    print(f"\nRegistering best model: {best_result['model_name']}")
    print(f"Source model URI: {best_result['model_uri']}")

    registered_model = mlflow.register_model(
        model_uri=best_result["model_uri"],
        name=REGISTERED_MODEL_NAME,
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered_model.version,
    )

    print(
        f"Registered model '{REGISTERED_MODEL_NAME}' "
        f"version {registered_model.version} with alias 'champion'"
    )


def main() -> None:
    data_path = Path("data/raw/Telco-Customer-Churn.csv")

    df = load_data(data_path)
    print("Raw shape:", df.shape)

    validate_raw_data(df)

    df_clean = clean_data(df)
    print("Cleaned shape:", df_clean.shape)

    X, y = split_features_target(df_clean)
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nFeature columns:")
    print(list(X.columns))

    preprocessor = build_preprocessor()
    print("\nPreprocessor created successfully:")
    print(preprocessor)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain/test split:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    print("\nStarting training...")

    models = get_models()

    mlflow.set_experiment("telco-churn")

    results = []

    for model_name, model in models.items():
        result = train_and_log_model(
            model_name,
            model,
            preprocessor,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        results.append(result)

    best_result = max(results, key=lambda r: r["metrics"]["f1"])

    print("\nBest model selected:")
    print(f"Model: {best_result['model_name']}")
    print(f"F1 score: {best_result['metrics']['f1']:.4f}")

    register_best_model(best_result)

if __name__ == "__main__":
    main()