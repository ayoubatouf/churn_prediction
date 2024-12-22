from pathlib import Path
from typing import List, Optional, Tuple
import joblib
import copy
import pandas as pd
from src.data.preprocess_data import clean_data, encode_data
from src.features.build_features import feature_engineering
from src.models.predict import predict_on_test_data
from src.utils.file_io import write_inference_to_file
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def inference_pipeline_(
    data_path: Path,
    model_path: Path,
    scaler_path: Path,
    output_path: Path,
    numerical_cols: Optional[List[str]] = ["TotalCharges"],
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:

    try:
        logging.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        logging.info(f"Loading scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)

        logging.info(f"Loading inference data from {data_path}...")
        data = pd.read_csv(data_path)

        input_data = copy.deepcopy(data)

        logging.info("Cleaning and preparing the data for inference...")
        data = clean_data(data)
        data = encode_data(data)
        data = feature_engineering(data)
        print("data feature engineered")

        if numerical_cols:
            logging.info(f"Scaling numerical columns: {numerical_cols}...")
            data[numerical_cols] = scaler.transform(data[numerical_cols])

        X = data.drop(columns="Churn", errors="ignore")

        logging.info("Making predictions...")
        predictions, prediction_proba = predict_on_test_data(model, X)

        write_inference_to_file(input_data, predictions, prediction_proba, output_path)

        logging.info("Inference completed successfully.")
        return predictions, prediction_proba

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return None, None
    except pd.errors.EmptyDataError:
        logging.error(f"No data found in file: {data_path}")
        return None, None
    except pd.errors.ParserError:
        logging.error(f"Error parsing the data from file: {data_path}")
        return None, None
    except Exception as e:
        logging.error(f"An error occurred during inference: {e}")
        return None, None
