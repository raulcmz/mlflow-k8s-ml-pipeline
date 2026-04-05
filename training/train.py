from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def main() -> None:
    data_path = Path("data/raw/Telco-Customer-Churn.csv")

    df = load_data(data_path)
    print("Raw shape:", df.shape)

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


if __name__ == "__main__":
    main()