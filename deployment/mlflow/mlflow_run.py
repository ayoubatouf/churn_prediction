import sys
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from config.paths import MODEL_PATH, PROCESSED_DATA_PATH
from src.models.evaluate import evaluate_model
from src.models.predict import predict_on_test_data

with mlflow.start_run():

    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(PROCESSED_DATA_PATH)

    print("Data info before conversion:")
    print(data.info())
    data = data.astype(
        {col: "float" for col in data.select_dtypes(include="int").columns}
    )

    if data.isnull().any().any():
        raise ValueError(
            "Data contains missing values. Please handle them before logging the model."
        )

    print("Data types after conversion:", data.dtypes)

    target_column = "Churn"
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the data.")

    X, y = data.drop(columns=[target_column]).astype(float), data[target_column].astype(
        float
    )

    if y.nunique() == 2 and set(y.unique()).issubset({0, 1}):
        min_class_size = y.value_counts().min()
        X_test = pd.concat(
            [X[y == 0].sample(n=min_class_size, random_state=42), X[y == 1]]
        )
        y_test = pd.concat(
            [y[y == 0].sample(n=min_class_size, random_state=42), y[y == 1]]
        )
    else:
        raise ValueError(
            "Target variable is not binary or not properly labeled as 0 and 1."
        )

    X_test, y_test = X_test.sample(frac=1, random_state=42).reset_index(
        drop=True, inplace=False
    ), y_test.sample(frac=1, random_state=42).reset_index(drop=True, inplace=False)

    y_pred, y_pred_proba = predict_on_test_data(model, X_test)

    results = evaluate_model(y_test, y_pred, y_pred_proba)

    mlflow.log_metrics(
        {
            "Accuracy": results["Accuracy"],
            "F1 Score": results["F1 Score"],
            "ROC-AUC": results["ROC-AUC"],
            "Precision": results["Precision"],
            "Recall": results["Recall"],
            "MCC": results["MCC"],
            "Log Loss": results["Log Loss"],
        }
    )

    if hasattr(model, "get_params"):
        mlflow.log_params(model.get_params())

    mlflow.sklearn.log_model(model, "model", input_example=X_test.iloc[0:1])

    mlflow.set_tags(
        {
            "project_name": "Churn Prediction",
            "experiment_id": "exp_1",
            "data_version": "1.0",
            "python_version": sys.version,
        }
    )

    sample_predictions = pd.DataFrame(
        {
            "true_labels": y_test,
            "predicted_labels": y_pred,
            "predicted_probabilities": y_pred_proba,
        }
    )

    print("Evaluation results logged in MLflow.")
