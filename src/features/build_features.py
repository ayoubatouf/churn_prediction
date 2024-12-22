import pandas as pd
from config.config import FEATURES_JSON_PATH
from src.utils.file_io import load_config


feature_config = load_config(FEATURES_JSON_PATH)


def feature_engineering(data: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:

    features = feature_config["features"]

    if target in data.columns:
        data = data[features + [target]]
    else:
        data = data[features]

    return data
