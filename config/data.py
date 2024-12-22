from pathlib import Path
import os

PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent

REQUIRED_COLUMNS_JSON_PATH = Path(
    os.path.join(PROJECT_ROOT, "src", "data", "required_columns.json")
)
DATA_PROCESSING_JSON_PATH = Path(
    os.path.join(PROJECT_ROOT, "src", "data", "data_processing_config.json")
)
FEATURES_JSON_PATH = Path(
    os.path.join(PROJECT_ROOT, "src", "features", "features.json")
)
SEARCH_SPACE_JSON_PATH = Path(
    os.path.join(PROJECT_ROOT, "src", "models", "search_space.json")
)
PREDICTION_THRESHOLD = 0.5
NUMERICAL_COLS = ["TotalCharges"]
