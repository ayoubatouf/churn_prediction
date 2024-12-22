from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from src.models.evaluate import evaluate_model
from src.models.model_definition import bayesian_search_ada
from src.models.predict import predict_on_test_data
from src.models.train import train_best_model
from src.utils.file_io import write_evaluation_to_file
from src.utils.metrics import plot_metrics
import joblib
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def training_pipeline_(
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    eval_results_path: Path,
    model_save_path: Path,
) -> Optional[ClassifierMixin]:

    try:
        logging.info("Starting Bayesian search for the best model...")
        best_model = bayesian_search_ada(X_train_scaled, y_train)

        logging.info("Training the best model...")
        trained_model = train_best_model(best_model, X_train_scaled, y_train)

        logging.info("Predicting on the test data...")
        y_pred, y_pred_proba = predict_on_test_data(trained_model, X_test_scaled)

        logging.info("Evaluating the model...")
        results = evaluate_model(y_test, y_pred, y_pred_proba)

        logging.info(f"Writing evaluation results to {eval_results_path}...")
        write_evaluation_to_file(
            results,
            trained_model.get_params(),
            filename=eval_results_path / "evaluation.txt",
        )

        logging.info(f"Plotting metrics and saving to {eval_results_path}...")
        plot_metrics(y_test, y_pred, y_pred_proba, save_path=eval_results_path)

        logging.info(f"Saving the trained model to {model_save_path}...")
        joblib.dump(trained_model, model_save_path)

        logging.info("Training pipeline completed successfully.")
        return trained_model

    except Exception as e:
        logging.error(f"An error occurred in the training pipeline: {e}")
        return None
