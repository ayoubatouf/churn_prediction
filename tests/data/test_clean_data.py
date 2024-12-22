import unittest
import pandas as pd
from unittest.mock import patch
from src.data.preprocess_data import clean_data


class TestCleanData(unittest.TestCase):

    @patch.dict(
        "src.data.preprocess_data.config",
        {
            "cleaning": {
                "replace_values": {"old_value": "new_value"},
                "drop_columns": ["col_to_drop"],
            }
        },
    )
    def test_clean_data(self):

        data = pd.DataFrame(
            {
                "col1": ["old_value", "other_value", "old_value"],
                "col2": [1, 2, 3],
                "col_to_drop": [10, 20, 30],
            }
        )

        cleaned_data = clean_data(data)
        self.assertTrue(
            (cleaned_data["col1"] == ["new_value", "other_value", "new_value"]).all(),
            "Replacement of old_value with new_value failed.",
        )

        self.assertNotIn(
            "col_to_drop",
            cleaned_data.columns,
            "Column 'col_to_drop' was not removed correctly.",
        )

        self.assertTrue(
            "col2" in cleaned_data.columns, "Column 'col2' was mistakenly dropped."
        )

    @patch.dict(
        "src.data.preprocess_data.config",
        {"cleaning": {"replace_values": {}, "drop_columns": ["non_existent_col"]}},
    )
    def test_clean_data_no_changes(self):

        data = pd.DataFrame({"col1": ["value1", "value2", "value3"], "col2": [4, 5, 6]})

        cleaned_data = clean_data(data)

        self.assertTrue(
            (cleaned_data["col1"] == ["value1", "value2", "value3"]).all(),
            "Data in col1 was unexpectedly modified.",
        )
        self.assertTrue(
            (cleaned_data["col2"] == [4, 5, 6]).all(),
            "Data in col2 was unexpectedly modified.",
        )

    @patch.dict(
        "src.data.preprocess_data.config",
        {
            "cleaning": {
                "replace_values": {"invalid_value": "new_value"},
                "drop_columns": [],
            }
        },
    )
    def test_clean_data_no_replacement_found(self):
        data = pd.DataFrame({"col1": ["value1", "value2", "value3"], "col2": [7, 8, 9]})

        cleaned_data = clean_data(data)

        self.assertTrue(
            (cleaned_data["col1"] == ["value1", "value2", "value3"]).all(),
            "Unexpected replacement occurred in col1.",
        )
        self.assertTrue(
            (cleaned_data["col2"] == [7, 8, 9]).all(),
            "Unexpected replacement occurred in col2.",
        )

    @patch.dict(
        "src.data.preprocess_data.config",
        {"cleaning": {"replace_values": {}, "drop_columns": ["col1", "col2"]}},
    )
    def test_clean_data_column_drop(self):

        data = pd.DataFrame(
            {"col1": ["val1", "val2", "val3"], "col2": [1, 2, 3], "col3": [4, 5, 6]}
        )

        cleaned_data = clean_data(data)

        self.assertNotIn("col1", cleaned_data.columns, "Column 'col1' was not dropped.")
        self.assertNotIn("col2", cleaned_data.columns, "Column 'col2' was not dropped.")
        self.assertIn(
            "col3", cleaned_data.columns, "Column 'col3' was mistakenly dropped."
        )


if __name__ == "__main__":
    unittest.main()
