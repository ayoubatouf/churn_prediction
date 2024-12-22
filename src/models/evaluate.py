from typing import Dict
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    classification_report,
)


def evaluate_model(
    y_test: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series
) -> Dict[str, object]:

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    confusion = confusion_matrix(y_test, y_pred)
    classification_report_str = classification_report(y_test, y_pred)

    results = {
        "Accuracy": accuracy,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "MCC": mcc,
        "Log Loss": logloss,
        "Confusion Matrix": confusion,
        "Classification Report": classification_report_str,
    }

    return results
