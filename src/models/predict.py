from typing import Tuple
import pandas as pd
from sklearn.base import ClassifierMixin


def predict_on_test_data(
    model: ClassifierMixin, X_test: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:

    if hasattr(model, "predict_proba"):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        raise AttributeError(
            "The model does not support probability predictions (predict_proba)."
        )

    return y_pred, y_pred_proba
