from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from src.data.loading_data import load_data, validate_data_columns
from src.data.preprocess_data import (
    balance_data,
    clean_data,
    encode_data,
    scale_numerical_features,
    split_data,
)
from src.features.build_features import feature_engineering
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def data_pipeline_(
    file_path: Path, output_path: Path, scaler_path: Path
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:

    try:
        data = load_data(file_path)

        if not validate_data_columns(data):
            logging.error("Data validation failed: required columns are missing.")
            return None

        data = clean_data(data)
        data = encode_data(data)
        data = feature_engineering(data)

        data.to_csv(path_or_buf=output_path, index=False)
        logging.info(f"Data processed and saved to {output_path}.")

        data = balance_data(data)

        X_train, X_test, y_train, y_test = split_data(data)
        X_train_scaled, X_test_scaled = scale_numerical_features(
            X_train, X_test, scaler_path, save_scaler=True
        )

        return X_train_scaled, X_test_scaled, y_train, y_test

    except FileNotFoundError:
        logging.error(
            f"File not found: {file_path}. Please check the path and try again."
        )
    except Exception as e:
        logging.error(f"An error occurred during the data pipeline: {e}")

    return None
