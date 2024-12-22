import unittest
from unittest.mock import patch
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from src.data.preprocess_data import scale_numerical_features


class TestScaleNumericalFeatures(unittest.TestCase):

    def setUp(self):
        self.X_train = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]})
        self.X_test = pd.DataFrame({"A": [2, 3, 4], "B": [6, 7, 8]})
        self.scaler_path = Path("scaler.pkl")
        self.numerical_cols = ["A", "B"]

    @patch("joblib.dump")
    def test_scale_numerical_features(self, mock_dump):
        X_train_scaled, X_test_scaled = scale_numerical_features(
            self.X_train,
            self.X_test,
            self.scaler_path,
            numerical_cols=self.numerical_cols,
            save_scaler=True,
        )

        mock_dump.assert_called_once()

        scaler = MinMaxScaler()
        scaler.fit(self.X_train[self.numerical_cols])

        pd.testing.assert_frame_equal(
            X_train_scaled,
            pd.DataFrame(
                scaler.transform(self.X_train[self.numerical_cols]),
                columns=self.numerical_cols,
            ),
        )
        pd.testing.assert_frame_equal(
            X_test_scaled,
            pd.DataFrame(
                scaler.transform(self.X_test[self.numerical_cols]),
                columns=self.numerical_cols,
            ),
        )

    @patch("joblib.dump")
    def test_no_scaling_saved(self, mock_dump):
        X_train_scaled, X_test_scaled = scale_numerical_features(
            self.X_train,
            self.X_test,
            self.scaler_path,
            numerical_cols=self.numerical_cols,
            save_scaler=False,
        )
        mock_dump.assert_not_called()

        scaler = MinMaxScaler()
        scaler.fit(self.X_train[self.numerical_cols])

        pd.testing.assert_frame_equal(
            X_train_scaled,
            pd.DataFrame(
                scaler.transform(self.X_train[self.numerical_cols]),
                columns=self.numerical_cols,
            ),
        )
        pd.testing.assert_frame_equal(
            X_test_scaled,
            pd.DataFrame(
                scaler.transform(self.X_test[self.numerical_cols]),
                columns=self.numerical_cols,
            ),
        )


if __name__ == "__main__":
    unittest.main()
