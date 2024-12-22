import os
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def plot_roc_curve(
    y_test: np.ndarray, y_pred_proba: np.ndarray, save_path: Optional[str] = None
) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc_value = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc_value:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()

    if save_path:
        if os.path.isdir(save_path):
            file_name = "roc_curve.png"
            full_path = os.path.join(save_path, file_name)
        else:
            full_path = save_path

        plt.savefig(full_path)
        print(f"ROC Curve saved to {full_path}")
    else:
        plt.show()


def plot_precision_recall_curve(
    y_test: np.ndarray, y_pred_proba: np.ndarray, save_path: Optional[str] = None
) -> None:
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    average_precision = average_precision_score(y_test, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(
        recall,
        precision,
        color="blue",
        label=f"Precision-Recall Curve (AP = {average_precision:.2f})",
    )
    plt.fill_between(recall, precision, alpha=0.2, color="blue")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid()

    if save_path:
        if os.path.isdir(save_path):
            file_name = "precision_recall_curve.png"
            full_path = os.path.join(save_path, file_name)
        else:
            full_path = save_path

        plt.savefig(full_path)
        print(f"Precision-Recall Curve saved to {full_path}")
    else:
        plt.show()


def plot_confusion_matrix(
    y_test: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None
) -> None:
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No Churn", "Churn"]
    )
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    if save_path:
        if os.path.isdir(save_path):
            file_name = "confusion_matrix.png"
            full_path = os.path.join(save_path, file_name)
        else:
            full_path = save_path

        plt.savefig(full_path)
        print(f"Confusion Matrix saved to {full_path}")
    else:
        plt.show()


def plot_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    plot_roc_curve(y_test, y_pred_proba, save_path)
    plot_precision_recall_curve(y_test, y_pred_proba, save_path)
    plot_confusion_matrix(y_test, y_pred, save_path)
