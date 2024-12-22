import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from src.models.predict import predict_on_test_data


class TestPredictOnTestData(unittest.TestCase):

    def setUp(self):
        self.X_test = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        self.mock_model = MagicMock()

    def test_predict_on_test_data_with_predict_proba(self):

        self.mock_model.predict.return_value = pd.Series([0, 1, 1])

        self.mock_model.predict_proba.return_value = np.array(
            [[0.2, 0.8], [0.1, 0.9], [0.4, 0.6]]
        )

        y_pred, y_pred_proba = predict_on_test_data(self.mock_model, self.X_test)

        self.mock_model.predict.assert_called_once_with(self.X_test)
        self.mock_model.predict_proba.assert_called_once_with(self.X_test)

        self.assertTrue((y_pred == pd.Series([0, 1, 1])).all())
        self.assertTrue((y_pred_proba == pd.Series([0.8, 0.9, 0.6])).all())

    def test_predict_on_test_data_without_predict_proba(self):

        self.mock_model.predict.return_value = pd.Series([0, 1, 0])
        del self.mock_model.predict_proba

        with self.assertRaises(AttributeError):
            predict_on_test_data(self.mock_model, self.X_test)


if __name__ == "__main__":
    unittest.main()
