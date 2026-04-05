from pathlib import Path

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load the raw dataset from a CSV file.

    Args:
        path: Path to the raw CSV file.

    Returns:
        pandas.DataFrame with the raw data.
    """
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Telco Churn dataset.

    Steps:
    - Drop customerID because it is only an identifier
    - Convert TotalCharges to numeric
    - Drop rows with missing TotalCharges
    - Convert Churn to binary target: Yes -> 1, No -> 0

    Args:
        df: Raw input DataFrame.

    Returns:
        Cleaned DataFrame ready for feature/target split.
    """
    df = df.copy()

    # Drop identifier column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric, invalid values become NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing values in TotalCharges
    df = df.dropna(subset=["TotalCharges"])

    # Convert target column to binary
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


def main() -> None:
    data_path = Path("data/raw/Telco-Customer-Churn.csv")

    df = load_data(data_path)
    print("Raw shape:", df.shape)
    print(df.head())

    df_clean = clean_data(df)
    print("Cleaned shape:", df_clean.shape)
    print(df_clean.head())


if __name__ == "__main__":
    main()