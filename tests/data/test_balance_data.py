import unittest
import pandas as pd
from src.data.preprocess_data import balance_data


class TestBalanceData(unittest.TestCase):

    def setUp(self):
        self.sample_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 6],
                "feature2": [7, 8, 9, 10, 11, 12],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )

    def test_balance_data(self):
        balanced_data = balance_data(
            self.sample_data, target="target", n_sample=2, random_seed=42
        )

        target_counts = balanced_data["target"].value_counts()

        self.assertEqual(target_counts[1], 3)
        self.assertEqual(target_counts[0], 2)

        self.assertEqual(len(balanced_data), 5)

    def test_sampling_limit(self):

        n_sample = 5
        num_target_0 = len(self.sample_data[self.sample_data["target"] == 0])
        n_sample = min(n_sample, num_target_0)
        balanced_data = balance_data(
            self.sample_data, target="target", n_sample=n_sample, random_seed=42
        )

        target_counts = balanced_data["target"].value_counts()

        self.assertEqual(target_counts[0], 3)
        self.assertEqual(target_counts[1], 3)

        self.assertEqual(len(balanced_data), 6)


if __name__ == "__main__":
    unittest.main()
