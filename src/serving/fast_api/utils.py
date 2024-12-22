from pathlib import Path
import time
from typing import Any, Dict, List
import joblib
import pandas as pd
import psutil
from pydantic import BaseModel
from config.data import PREDICTION_THRESHOLD
from config.paths import PROD_LOG_PATH
from src.data.preprocess_data import clean_data, encode_data
from src.features.build_features import feature_engineering
import logging
from config.config import MODEL_PATH, SCALER_PATH, NUMERICAL_COLS
from src.models.predict import predict_on_test_data
from datetime import datetime as dt


try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
    model, scaler = None, None


def preprocess_and_scale_data(data: pd.DataFrame) -> pd.DataFrame:

    data = clean_data(data)
    data = encode_data(data)
    data = feature_engineering(data)
    data.drop(columns="Churn", errors="ignore")
    if NUMERICAL_COLS:
        data[NUMERICAL_COLS] = scaler.transform(data[NUMERICAL_COLS])
    return data


class InputData(BaseModel):
    data: list[dict]


def append_to_csv(log_entries: List[Dict[str, Any]], file_path: Path) -> None:

    pd.DataFrame(log_entries).to_csv(file_path, mode="a", header=False, index=False)


def get_model_info(model: Any) -> tuple[str, Dict[str, Any]]:
    model_name = model.__class__.__name__
    model_params = model.get_params() if hasattr(model, "get_params") else {}
    return model_name, model_params


def initialize_log_file() -> None:

    try:
        pd.read_csv(PROD_LOG_PATH)
    except FileNotFoundError:
        pd.DataFrame(
            columns=[
                "input_data",
                "prediction",
                "probability",
                "date",
                "model_name",
                "model_params",
                "response_time",
                "memory_usage",
            ]
        ).to_csv(PROD_LOG_PATH, index=False)


async def process_prediction(
    input_data: InputData, start_time: float
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = pd.DataFrame.from_records(input_data.data)
    if data.empty:
        raise ValueError("Uploaded data is empty.")

    X = preprocess_and_scale_data(data)
    logging.info(f"Input data shape: {X.shape}")

    _, prediction_proba = predict_on_test_data(model, X)
    results = [
        {
            "will_leave": bool(proba > PREDICTION_THRESHOLD),
            "probability": float(proba),
        }
        for proba in prediction_proba
    ]

    model_name, model_params = get_model_info(model)
    current_time = dt.now().isoformat()
    response_time = time.time() - start_time
    final_memory = psutil.virtual_memory().percent

    log_entries = create_log_entries(
        data,
        results,
        model_name,
        model_params,
        current_time,
        response_time,
        final_memory,
    )

    return results, log_entries


def create_log_entries(
    data: pd.DataFrame,
    results: List[Dict[str, Any]],
    model_name: str,
    model_params: Dict[str, Any],
    current_time: str,
    response_time: float,
    final_memory: float,
) -> List[Dict[str, Any]]:
    return [
        {
            "input_data": row.to_dict(),
            "prediction": result["will_leave"],
            "probability": result["probability"],
            "date": current_time,
            "model_name": model_name,
            "model_params": model_params,
            "response_time": response_time,
            "memory_usage": final_memory,
        }
        for (_, row), result in zip(data.iterrows(), results)
    ]
