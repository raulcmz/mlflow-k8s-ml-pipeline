from pathlib import Path

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

from training.train import load_data, clean_data


REPORTS_DIR = Path("reports/evidently")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


NUMERICAL_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS = [
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



def prepare_reference_and_current(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the cleaned dataset into reference/current subsets
    and inject small controlled changes into the current dataset
    so the drift report becomes meaningful for the POC.
    """
    reference = df.sample(frac=0.5, random_state=42).copy()
    current = df.drop(reference.index).copy()

    # Inject controlled shifts for demonstration purposes
    current["MonthlyCharges"] = current["MonthlyCharges"] * 1.15
    current["tenure"] = (current["tenure"] * 0.85).clip(lower=0)
    current.loc[current.sample(frac=0.15, random_state=42).index, "Contract"] = "Month-to-month"

    return reference, current


def build_data_definition() -> DataDefinition:
    return DataDefinition(
        numerical_columns=NUMERICAL_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
    )


def generate_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Path:
    data_definition = build_data_definition()

    reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)

    report = Report([
        DataDriftPreset(),
    ])

    evaluation = report.run(current_dataset, reference_dataset)

    output_path = REPORTS_DIR / "data_drift_report.html"
    evaluation.save_html(str(output_path))

    return output_path


def main() -> None:
    data_path = Path("data/raw/Telco-Customer-Churn.csv")

    df = load_data(data_path)
    df_clean = clean_data(df)

    reference_df, current_df = prepare_reference_and_current(df_clean)
    report_path = generate_drift_report(reference_df, current_df)

    print(f"Evidently drift report generated at: {report_path}")


if __name__ == "__main__":
    main()