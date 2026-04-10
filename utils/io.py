import os

import yaml


def load_yaml(file_path: str) -> dict:
    """
    Load data from a YAML file.

    Args:
       file_path (str): path to the YAML file.

    Raise:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is not formatted correctly.

    Returns:
        data (dict): loaded data dictionary.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is empty or not formatted correctly.
    """
    # Verify that folder exists
    if not os.path.exists(file_path):
        raise FileNotFoundError

    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
        data_type = type(data)
        if data is None:
            raise ValueError(f"[IO] The file {file_path} is  empty.")
        elif not data_type == dict:
            raise ValueError(f"[IO] The file {file_path} is not formatted correctly.")

        # Return the loaded data
        return data
