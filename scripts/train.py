from config.config import (
    EVALUATION_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    SCALER_PATH,
)
from src.pipelines.data_pipeline import data_pipeline_
from src.pipelines.training_pipeline import training_pipeline_
import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_train_script(
    file_path=RAW_DATA_PATH,
    output_path=PROCESSED_DATA_PATH,
    eval_results_path=EVALUATION_PATH,
    model_save_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
):

    try:

        if not file_path.is_file():
            logging.error(f"The file at {file_path} was not found.")
            return None

        logging.info("Starting data processing...")
        data_pipeline_result = data_pipeline_(file_path, output_path, scaler_path)

        if data_pipeline_result is None:
            logging.error("Data pipeline failed. Exiting.")
            return None

        X_train_scaled, X_test_scaled, y_train, y_test = data_pipeline_result

        logging.info("Starting model training...")
        model = training_pipeline_(
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            eval_results_path,
            model_save_path,
        )

        logging.info("Model training complete and saved.")
        return model

    except Exception as e:
        logging.error(f"An error occurred during the training script execution: {e}")
        return None


if __name__ == "__main__":

    model = run_train_script()
    if model is not None:
        logging.info("Model training completed successfully.")
    else:
        logging.error("Model training failed.")
