import unittest
import pandas as pd
from src.data.preprocess_data import split_data


class TestSplitData(unittest.TestCase):

    def setUp(self):
        self.sample_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "feature2": [7, 8, 9, 10, 11, 12],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )

    def test_split_data(self):
        X_train, X_test, y_train, y_test = split_data(
            self.sample_data, target="target", test_size=0.33, random_seed=42
        )

        self.assertEqual(len(X_train), 4)
        self.assertEqual(len(X_test), 2)

        self.assertEqual(y_train.value_counts().to_dict(), {0: 2, 1: 2})
        self.assertEqual(y_test.value_counts().to_dict(), {0: 1, 1: 1})

        self.assertNotIn("target", X_train.columns)
        self.assertNotIn("target", X_test.columns)

    def test_random_seed(self):
        X_train1, X_test1, y_train1, y_test1 = split_data(
            self.sample_data, target="target", test_size=0.33, random_seed=42
        )
        X_train2, X_test2, y_train2, y_test2 = split_data(
            self.sample_data, target="target", test_size=0.33, random_seed=42
        )

        self.assertTrue(X_train1.equals(X_train2))
        self.assertTrue(X_test1.equals(X_test2))
        self.assertTrue(y_train1.equals(y_train2))
        self.assertTrue(y_test1.equals(y_test2))


if __name__ == "__main__":
    unittest.main()
