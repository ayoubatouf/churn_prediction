import unittest
from unittest.mock import patch
import pandas as pd
from src.data.loading_data import validate_data_columns


class TestValidateDataColumns(unittest.TestCase):

    @patch("src.utils.file_io.load_config")
    def test_validate_data_columns_all_columns_present(self, mock_load_config):
        mock_load_config.return_value = {
            "required_columns": [
                "customerID",
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "tenure",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "MonthlyCharges",
                "TotalCharges",
                "Churn",
            ]
        }

        data = pd.DataFrame(
            {
                "customerID": [1, 2],
                "gender": ["Male", "Female"],
                "SeniorCitizen": [0, 1],
                "Partner": ["Yes", "No"],
                "Dependents": ["No", "Yes"],
                "tenure": [12, 24],
                "PhoneService": ["Yes", "No"],
                "MultipleLines": ["No", "Yes"],
                "InternetService": ["Fiber optic", "DSL"],
                "OnlineSecurity": ["No", "Yes"],
                "OnlineBackup": ["Yes", "No"],
                "DeviceProtection": ["No", "Yes"],
                "TechSupport": ["Yes", "No"],
                "StreamingTV": ["No", "Yes"],
                "StreamingMovies": ["Yes", "No"],
                "Contract": ["Month-to-month", "One year"],
                "PaperlessBilling": ["Yes", "No"],
                "PaymentMethod": ["Electronic check", "Mailed check"],
                "MonthlyCharges": [50.99, 60.99],
                "TotalCharges": [600.99, 720.99],
                "Churn": ["Yes", "No"],
            }
        )

        is_valid, missing_columns = validate_data_columns(data)

        self.assertTrue(is_valid)
        self.assertEqual(missing_columns, [])

    @patch("src.utils.file_io.load_config")
    def test_validate_data_columns_missing_columns(self, mock_load_config):
        mock_load_config.return_value = {
            "required_columns": [
                "customerID",
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "tenure",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "MonthlyCharges",
                "TotalCharges",
                "Churn",
            ]
        }

        data = pd.DataFrame(
            {
                "customerID": [1, 2],
                "SeniorCitizen": [0, 1],
                "Partner": ["Yes", "No"],
                "Dependents": ["No", "Yes"],
                "tenure": [12, 24],
                "PhoneService": ["Yes", "No"],
            }
        )

        is_valid, missing_columns = validate_data_columns(data)

        expected_missing_columns = [
            "gender",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
            "Churn",
        ]

        self.assertFalse(is_valid)
        self.assertEqual(sorted(missing_columns), sorted(expected_missing_columns))


if __name__ == "__main__":
    unittest.main()
