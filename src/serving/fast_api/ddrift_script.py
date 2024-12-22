import ast
import pandas as pd
from config.paths import PROD_LOG_PATH, RAW_DATA_PATH
from src.utils.data_drift import detect_drift

if __name__ == "__main__":

    columns_of_interest = [
        "Contract",
        "PaymentMethod",
        "InternetService",
        "TotalCharges",
        "TechSupport",
        "PaperlessBilling",
        "Dependents",
        "SeniorCitizen",
        "PhoneService",
        "gender",
    ]

    df1 = pd.read_csv(RAW_DATA_PATH, usecols=columns_of_interest)

    df2_raw = pd.read_csv(PROD_LOG_PATH, usecols=["input_data"])

    # extract fields from input_data column
    df2_expanded = df2_raw["input_data"].apply(lambda x: pd.Series(ast.literal_eval(x)))

    df2 = df2_expanded[columns_of_interest]

    drift_exists = detect_drift(df1, df2, columns_of_interest)
    print("Drift detected across any column:", drift_exists)
