from pathlib import Path
import pandas as pd
from config.config import REQUIRED_COLUMNS_JSON_PATH
from src.utils.file_io import load_config
from typing import List, Optional, Tuple

config = load_config(REQUIRED_COLUMNS_JSON_PATH)


def load_data(file_path: Path) -> Optional[pd.DataFrame]:
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None


def validate_data_columns(
    data: pd.DataFrame, required_columns: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:

    if required_columns is None:
        required_columns = config.get("required_columns", [])

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Warning: Missing columns - {missing_columns}")
        return False, missing_columns
    return True, []
