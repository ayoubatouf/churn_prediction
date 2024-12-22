from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from config.config import DATA_PROCESSING_JSON_PATH
from src.utils.file_io import load_config


config = load_config(DATA_PROCESSING_JSON_PATH)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:

    data.replace(config["cleaning"]["replace_values"], inplace=True)
    data.drop(columns=config["cleaning"]["drop_columns"], inplace=True, errors="ignore")
    return data


def encode_data(data: pd.DataFrame) -> pd.DataFrame:

    data = pd.get_dummies(
        data, columns=["gender", "PaymentMethod", "InternetService"], drop_first=False
    )

    binary_columns = config["encoding"]["binary_columns"]

    if "Churn" in data.columns:
        data["Churn"] = data["Churn"].replace(config["encoding"]["churn_mapping"])

    data[binary_columns] = data[binary_columns].replace({"Yes": 1, "No": 0})

    contract_mapping = config["encoding"]["contract_mapping"]
    data["Contract"] = data["Contract"].map(contract_mapping)

    data["TotalCharges"] = data["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    totalcharges_mean = data["TotalCharges"].mean()
    data["TotalCharges"].fillna(totalcharges_mean, inplace=True)

    data = data.replace({True: 1, False: 0})

    data.drop(
        columns=["gender_Female", "InternetService_No"], inplace=True, errors="ignore"
    )

    return data


def balance_data(
    data: pd.DataFrame,
    target: str = config["balancing"]["target"],
    n_sample: int = config["balancing"]["n_sample"],
    random_seed: int = config["balancing"]["random_seed"],
) -> pd.DataFrame:

    churn_1_data = data[data[target] == 1]
    churn_0_data = data[data[target] == 0].sample(n=n_sample, random_state=random_seed)

    data = pd.concat([churn_1_data, churn_0_data])
    data = shuffle(data, random_state=random_seed)

    return data


def split_data(
    data: pd.DataFrame,
    target: str = config["splitting"]["target"],
    test_size: float = config["splitting"]["test_size"],
    random_seed: int = config["splitting"]["random_seed"],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    return X_train, X_test, y_train, y_test


def scale_numerical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: Path,
    numerical_cols: Optional[list[str]] = config["scaling"]["numerical_cols"],
    save_scaler: bool = config["scaling"]["save_scaler"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    scaler = MinMaxScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    if save_scaler:
        joblib.dump(scaler, scaler_path)

    return X_train, X_test
