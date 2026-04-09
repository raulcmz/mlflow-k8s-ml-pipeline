from pathlib import Path
import os

import pandas as pd
import mlflow
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from training.train import load_data, clean_data


REPORTS_DIR = Path("reports/evidently")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "telco-churn"

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

    # Controlled shifts for demonstration purposes
    current["MonthlyCharges"] = current["MonthlyCharges"] * 1.15
    current["tenure"] = (current["tenure"] * 0.85).clip(lower=0)
    current.loc[current.sample(frac=0.15, random_state=42).index, "Contract"] = "Month-to-month"

    return reference, current


def build_data_definition() -> DataDefinition:
    return DataDefinition(
        numerical_columns=NUMERICAL_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
    )


def build_datasets(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> tuple[Dataset, Dataset]:
    data_definition = build_data_definition()

    reference_dataset = Dataset.from_pandas(
        reference_df,
        data_definition=data_definition,
    )
    current_dataset = Dataset.from_pandas(
        current_df,
        data_definition=data_definition,
    )

    return reference_dataset, current_dataset


def generate_data_drift_report(reference_dataset: Dataset, current_dataset: Dataset) -> tuple[Path, dict]:
    report = Report([
        DataDriftPreset(),
    ])

    evaluation = report.run(current_dataset, reference_dataset)

    output_path = REPORTS_DIR / "data_drift_report.html"
    evaluation.save_html(str(output_path))

    report_dict = evaluation.dict()

    metrics = {
        "drift_share": 0.0,
        "drifted_columns": 0,
        "total_columns": 0,
        "dataset_drift": 0,
    }

    try:
        metric_results = report_dict["metrics"][0]["result"]
        metrics["drift_share"] = float(metric_results.get("share_of_drifted_columns", 0.0))
        metrics["drifted_columns"] = int(metric_results.get("number_of_drifted_columns", 0))
        metrics["total_columns"] = int(metric_results.get("number_of_columns", 0))
        metrics["dataset_drift"] = int(bool(metric_results.get("dataset_drift", False)))
    except (KeyError, IndexError, TypeError, ValueError):
        print("Warning: could not extract structured drift metrics from Evidently report output.")

    return output_path, metrics


def generate_data_summary_report(reference_dataset: Dataset, current_dataset: Dataset) -> Path:
    report = Report([
        DataSummaryPreset(),
    ])

    evaluation = report.run(current_dataset, reference_dataset)

    output_path = REPORTS_DIR / "data_summary_report.html"
    evaluation.save_html(str(output_path))

    return output_path


def log_reports_to_mlflow(drift_report_path: Path, summary_report_path: Path, drift_metrics: dict) -> None:
    with mlflow.start_run(run_name="drift_report"):
        mlflow.log_artifact(str(drift_report_path), artifact_path="drift_reports")
        mlflow.log_artifact(str(summary_report_path), artifact_path="summary_reports")

        mlflow.log_metric("drift_share", drift_metrics["drift_share"])
        mlflow.log_metric("drifted_columns", drift_metrics["drifted_columns"])
        mlflow.log_metric("total_columns", drift_metrics["total_columns"])
        mlflow.log_metric("dataset_drift", drift_metrics["dataset_drift"])


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Using MLFLOW_TRACKING_URI={tracking_uri}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment:
        print(f"Using experiment: {experiment.name} (id={experiment.experiment_id})")

    data_path = Path("data/raw/Telco-Customer-Churn.csv")

    df = load_data(data_path)
    df_clean = clean_data(df)

    reference_df, current_df = prepare_reference_and_current(df_clean)
    reference_dataset, current_dataset = build_datasets(reference_df, current_df)

    drift_report_path, drift_metrics = generate_data_drift_report(reference_dataset, current_dataset)
    summary_report_path = generate_data_summary_report(reference_dataset, current_dataset)

    log_reports_to_mlflow(drift_report_path, summary_report_path, drift_metrics)

    print(f"Evidently drift report generated at: {drift_report_path}")
    print(f"Evidently summary report generated at: {summary_report_path}")
    print(f"Logged drift metrics: {drift_metrics}")


if __name__ == "__main__":
    main()