import unittest
from unittest.mock import patch
import pandas as pd
from src.features.build_features import feature_engineering


class TestFeatureEngineering(unittest.TestCase):

    @patch("src.utils.file_io.load_config")
    def test_feature_engineering_with_target_column(self, mock_load_config):

        mock_load_config.return_value = {
            "features": [
                "Contract",
                "PaymentMethod_Electronic check",
                "InternetService_Fiber optic",
                "TotalCharges",
                "TechSupport",
                "PaperlessBilling",
                "Dependents",
                "SeniorCitizen",
                "PhoneService",
                "gender_Male",
            ]
        }

        data = pd.DataFrame(
            {
                "Contract": ["Month-to-month", "One year", "Two year"],
                "PaymentMethod_Electronic check": ["Yes", "No", "Yes"],
                "InternetService_Fiber optic": ["Yes", "No", "Yes"],
                "TotalCharges": [100, 200, 150],
                "TechSupport": ["Yes", "No", "Yes"],
                "PaperlessBilling": ["Yes", "No", "Yes"],
                "Dependents": [1, 0, 1],
                "SeniorCitizen": [0, 1, 0],
                "PhoneService": ["Yes", "No", "Yes"],
                "gender_Male": [1, 0, 1],
                "Churn": [0, 1, 0],
            }
        )

        result = feature_engineering(data, target="Churn")

        expected_columns = [
            "Contract",
            "PaymentMethod_Electronic check",
            "InternetService_Fiber optic",
            "TotalCharges",
            "TechSupport",
            "PaperlessBilling",
            "Dependents",
            "SeniorCitizen",
            "PhoneService",
            "gender_Male",
            "Churn",
        ]
        self.assertListEqual(result.columns.tolist(), expected_columns)

    @patch("src.utils.file_io.load_config")
    def test_feature_engineering_without_target_column(self, mock_load_config):

        mock_load_config.return_value = {
            "features": [
                "Contract",
                "PaymentMethod_Electronic check",
                "InternetService_Fiber optic",
                "TotalCharges",
                "TechSupport",
                "PaperlessBilling",
                "Dependents",
                "SeniorCitizen",
                "PhoneService",
                "gender_Male",
            ]
        }

        data = pd.DataFrame(
            {
                "Contract": ["Month-to-month", "One year", "Two year"],
                "PaymentMethod_Electronic check": ["Yes", "No", "Yes"],
                "InternetService_Fiber optic": ["Yes", "No", "Yes"],
                "TotalCharges": [100, 200, 150],
                "TechSupport": ["Yes", "No", "Yes"],
                "PaperlessBilling": ["Yes", "No", "Yes"],
                "Dependents": [1, 0, 1],
                "SeniorCitizen": [0, 1, 0],
                "PhoneService": ["Yes", "No", "Yes"],
                "gender_Male": [1, 0, 1],
            }
        )

        result = feature_engineering(data, target="Churn")

        expected_columns = [
            "Contract",
            "PaymentMethod_Electronic check",
            "InternetService_Fiber optic",
            "TotalCharges",
            "TechSupport",
            "PaperlessBilling",
            "Dependents",
            "SeniorCitizen",
            "PhoneService",
            "gender_Male",
        ]
        self.assertListEqual(result.columns.tolist(), expected_columns)


if __name__ == "__main__":
    unittest.main()
