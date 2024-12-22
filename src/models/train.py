import pandas as pd
from sklearn.base import ClassifierMixin


def train_best_model(
    model: ClassifierMixin,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    verbose: bool = False,
) -> ClassifierMixin:

    if verbose:
        print("Training the model...")

    model.fit(X_train, y_train)

    if verbose:
        print("Model training completed.")

    return model
