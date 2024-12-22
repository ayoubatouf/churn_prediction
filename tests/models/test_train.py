import unittest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.models.train import train_best_model


class TestTrainBestModel(unittest.TestCase):

    def setUp(self):
        X, y = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)

    def test_train_best_model(self):
        trained_model = train_best_model(
            self.model, self.X_train, self.y_train, verbose=True
        )

        self.assertTrue(
            hasattr(trained_model, "predict"),
            "The model should have the 'predict' method after training.",
        )

        predictions = trained_model.predict(self.X_test)
        self.assertEqual(
            len(predictions),
            len(self.y_test),
            "The number of predictions should match the number of test samples.",
        )


if __name__ == "__main__":
    unittest.main()
