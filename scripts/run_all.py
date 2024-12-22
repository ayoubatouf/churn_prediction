from config.config import EXP_LOG_PATH, INFERECE_INPUT_PATH, RAW_DATA_PATH
from scripts.inference import run_inference_script
from scripts.train import run_train_script
import logging
from src.utils.logger import setup_logger

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_all_script():

    if not RAW_DATA_PATH.is_file():
        logging.error(
            f"The file at {RAW_DATA_PATH} was not found. Exiting pipeline execution."
        )
        return

    if not INFERECE_INPUT_PATH.is_file():
        logging.error(
            f"The file at {INFERECE_INPUT_PATH} was not found. Exiting pipeline execution."
        )
        return

    try:
        logging.info("Starting the training process...")
        run_train_script()
        logging.info("Training process completed.")

        logging.info("Starting the inference process...")
        run_inference_script()
        logging.info("Inference process completed.")

    except Exception as e:
        logging.error(f"An error occurred during the pipeline execution: {e}")


if __name__ == "__main__":
    setup_logger(EXP_LOG_PATH)
    run_all_script()
    print("Entire pipeline execution finished.")
