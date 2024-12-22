import unittest
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.models.evaluate import evaluate_model


class TestEvaluateModel(unittest.TestCase):

    def setUp(self):
        X, y = np.random.rand(100, 5), np.random.randint(0, 2, 100)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

    def test_evaluate_model(self):
        results = evaluate_model(self.y_test, self.y_pred, self.y_pred_proba)

        expected_keys = [
            "Accuracy",
            "F1 Score",
            "ROC-AUC",
            "Precision",
            "Recall",
            "MCC",
            "Log Loss",
            "Confusion Matrix",
            "Classification Report",
        ]
        for key in expected_keys:
            self.assertIn(key, results, f"Missing key in results: {key}")

        self.assertIsInstance(results["Accuracy"], float)
        self.assertIsInstance(results["F1 Score"], float)
        self.assertIsInstance(results["ROC-AUC"], float)
        self.assertIsInstance(results["Precision"], float)
        self.assertIsInstance(results["Recall"], float)
        self.assertIsInstance(results["MCC"], float)
        self.assertIsInstance(results["Log Loss"], float)
        self.assertIsInstance(results["Confusion Matrix"], np.ndarray)
        self.assertIsInstance(results["Classification Report"], str)

    def test_accuracy_score(self):
        results = evaluate_model(self.y_test, self.y_pred, self.y_pred_proba)
        expected_accuracy = accuracy_score(self.y_test, self.y_pred)
        self.assertAlmostEqual(
            results["Accuracy"],
            expected_accuracy,
            places=2,
            msg="Accuracy score is incorrect",
        )


if __name__ == "__main__":
    unittest.main()
