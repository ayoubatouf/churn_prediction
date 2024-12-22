import json

from config.config import INFERECE_INPUT_PATH_JSON
from config.paths import INFERECE_INPUT_PATH_FASTAPI


def transform_and_save_json(input_filepath, output_filepath):
    with open(input_filepath, "r") as file:
        column_based_json = json.load(file)

    keys = list(column_based_json.keys())
    num_records = len(column_based_json[keys[0]])

    row_based_data = [
        {key: column_based_json[key][i] for key in keys} for i in range(num_records)
    ]

    final_data = {"data": row_based_data}

    with open(output_filepath, "w") as file:
        json.dump(final_data, file, indent=4)


if __name__ == "__main__":
    transform_and_save_json(
        input_filepath=INFERECE_INPUT_PATH_JSON,
        output_filepath=INFERECE_INPUT_PATH_FASTAPI,
    )
