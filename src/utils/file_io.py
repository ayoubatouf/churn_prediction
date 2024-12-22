import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd


def load_config(json_file: str) -> Dict[str, Any]:
    try:
        with open(json_file, "r") as file:
            config = json.load(file)
            return config
    except FileNotFoundError:
        print(f"Error: The JSON file at {json_file} was not found.")
        return {}


def write_evaluation_to_file(
    results: Dict[str, Any], model_params: Dict[str, Any], filename: str
) -> None:
    with open(filename, "w") as f:
        f.write("Model Parameters:\n")
        for param, value in model_params.items():
            f.write(f"{param}: {value}\n")
        f.write("\n")

        for key, value in results.items():
            if isinstance(value, str):
                f.write(f"{key}:\n{value}\n\n")
            elif isinstance(value, np.ndarray):
                f.write(f"{key}: {np.mean(value):.2f}\n")
            else:
                f.write(f"{key}: {value:.2f}\n")

    print(f"Evaluation results written to {filename}")


def write_inference_to_file(
    data: pd.DataFrame,
    predictions: np.ndarray,
    prediction_proba: np.ndarray,
    output_path: Path,
) -> None:

    results_df = data.copy()
    results_df["Prediction"] = predictions
    results_df["Probability"] = prediction_proba

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path_or_buf=output_path, index=False)

    print(f"Inference results written to {output_path}!")


def load_data_from_json(json_file_path: Path) -> Dict[str, Any]:
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)
    return data
