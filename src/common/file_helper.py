import json
import os

class FileHelper:
    def __init__(self):
        pass

    def load_file_as_str(self, file_path: str) -> str:
        """
        Loads the file at a specificed path, returns an error otherwise
        
        :param file_path (str): The path to the file you want to load
        :return _file a str of the contents of the file
        """
        _file = ""
        try:
            with open(file_path, "r") as fp:
                _file = fp.read()
            return _file
        except Exception:
            raise Exception(f"Error reading file at: {file_path}")

    def load_file_as_json(self, file_path) -> dict:
        pass

    def save_to_json(self, save_path) -> None:
        pass