from typing import List
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


def detect_drift(
    df1: pd.DataFrame, df2: pd.DataFrame, columns: List[str], alpha: float = 0.05
) -> bool:

    drift_detected = False
    drift_columns = []

    for column in columns:

        if column not in df1.columns or column not in df2.columns:
            raise ValueError(f"Column '{column}' is not present in both datasets")

        if df1[column].dtype == "object":
            contingency_table = pd.crosstab(df1[column], df2[column])
            _, p_value, _, _ = chi2_contingency(contingency_table)
            print(f"Chi-Squared Test for {column}: p-value = {p_value}")
            if p_value < alpha:
                drift_detected = True
                drift_columns.append(column)

        elif pd.api.types.is_numeric_dtype(df1[column]):
            _, p_value = ks_2samp(df1[column], df2[column])
            print(f"K-S Test for {column}: p-value = {p_value}")
            if p_value < alpha:
                drift_detected = True
                drift_columns.append(column)

    if drift_columns:
        print("Drift detected in the following columns:", drift_columns)
    else:
        print("No drift detected in any columns.")

    return drift_detected
