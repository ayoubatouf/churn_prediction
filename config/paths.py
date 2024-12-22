import os
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent

MODEL_PATH = Path(
    os.path.join(PROJECT_ROOT, "results", "pretrained_models", "trained_model.joblib")
)
SCALER_PATH = Path(os.path.join(PROJECT_ROOT, "results", "scaler.pkl"))

PROD_LOG_PATH = Path(os.path.join(PROJECT_ROOT, "logs", "poduction_execution.csv"))
RAW_DATA_PATH = Path(os.path.join(PROJECT_ROOT, "data", "raw", "churn_raw.csv"))
PROCESSED_DATA_PATH = Path(
    os.path.join(PROJECT_ROOT, "data", "processed", "churn_processed.csv")
)
EVALUATION_PATH = Path(os.path.join(PROJECT_ROOT, "results", "evaluation_results"))

INFERECE_INPUT_PATH = Path(
    os.path.join(PROJECT_ROOT, "data", "inference", "churn_inference.csv")
)
INFERECE_INPUT_PATH_JSON = Path(
    os.path.join(PROJECT_ROOT, "data", "inference", "churn_inference.json")
)
INFERECE_INPUT_PATH_FASTAPI = Path(
    os.path.join(PROJECT_ROOT, "serving", "fast_api", "test", "fastapi_input.json")
)
INFERECE_OUTPUT_PATH = Path(
    os.path.join(PROJECT_ROOT, "results", "inference_results", "inference_results.csv")
)
EXP_LOG_PATH = Path(os.path.join(PROJECT_ROOT, "logs", "pipeline_execution.log"))
EDA_PATH = Path(os.path.join(PROJECT_ROOT, "reports", "figures"))
