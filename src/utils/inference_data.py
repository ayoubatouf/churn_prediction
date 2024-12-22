import pandas as pd
from config.config import INFERECE_INPUT_PATH, INFERECE_INPUT_PATH_JSON
from src.utils.file_io import load_data_from_json


def create_sample_data(
    file_path=INFERECE_INPUT_PATH,
    json_file_path=INFERECE_INPUT_PATH_JSON,
):

    try:
        json_file_path = json_file_path.resolve()
        data = load_data_from_json(json_file_path)

    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return

    sample_df = pd.DataFrame(data)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")


if __name__ == "__main__":
    create_sample_data()
