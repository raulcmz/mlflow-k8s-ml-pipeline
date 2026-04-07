import pandas as pd
import great_expectations as gx


REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def validate_raw_data(df: pd.DataFrame) -> None:
    """
    Validate the raw Telco Churn dataset using Great Expectations GX Core.

    Raises:
        ValueError: If one or more validations fail.
    """
    context = gx.get_context()

    data_source = context.data_sources.add_pandas("telco_pandas_source")
    data_asset = data_source.add_dataframe_asset(name="telco_dataframe_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "telco_batch_definition"
    )
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    expectations = [
        gx.expectations.ExpectTableColumnsToMatchSet(column_set=REQUIRED_COLUMNS),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="customerID"),
        gx.expectations.ExpectColumnValuesToBeUnique(column="customerID"),
        gx.expectations.ExpectColumnValuesToBeInSet(column="Churn", value_set=["Yes", "No"]),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="tenure"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"),
        gx.expectations.ExpectColumnValuesToBeBetween(column="tenure", min_value=0),
        gx.expectations.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0),
    ]

    failed_expectations = []

    for expectation in expectations:
        result = batch.validate(expectation)
        if not result.success:
            failed_expectations.append(expectation.type)

    if failed_expectations:
        raise ValueError(
            "Great Expectations validation failed. "
            f"Failed expectations: {failed_expectations}"
        )

    print("Great Expectations validation passed successfully.")