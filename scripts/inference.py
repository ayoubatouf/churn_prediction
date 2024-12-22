from config.config import (
    INFERECE_INPUT_PATH,
    INFERECE_OUTPUT_PATH,
    MODEL_PATH,
    SCALER_PATH,
)
from src.pipelines.inference_pipeline import inference_pipeline_
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_inference_script(
    input_data_path=INFERECE_INPUT_PATH,
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    output_data_path=INFERECE_OUTPUT_PATH,
):

    if not input_data_path.is_file():
        logging.error(f"The file at {input_data_path} was not found.")
        return None

    try:
        result = inference_pipeline_(
            input_data_path, model_path, scaler_path, output_data_path
        )
        logging.info(f"Inference result: {result}")
        return result

    except Exception as e:
        logging.error(f"An error occurred during the inference process: {e}")
        return None


if __name__ == "__main__":
    run_inference_script()
    print("Inference completed.")
