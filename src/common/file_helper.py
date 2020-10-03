import json
import os

class FileHelper:
    def __init__(self):
        pass

    def loadFile(self, file_path: str) -> str:
        """
        Loads the file at a specificed path, returns an error otherwise
        
        :param file_path (str): The path to the file you want to load
        :return _file a str of the contents of the file
        """
        try:
            _file = ""
            with open(file_path, "r") as fp:
                _file = fp.read()
            return _file
        except Exception:
            raise Exception(f"Error reading file at: {file_path}")

    def loadJSON(self, file_path) -> dict:
        try:
            with open(file_path) as json_data:
                data = json.load(json_data)
            return data
        except Exception:
            raise Exception(f"Could not open file located at: {file_path}")

    def saveJSON(self, data, save_path) -> None:
        try:
            with open(save_path, "w+") as fp:
                json.dump(data, fp, indent=4)
        except Exception:
            raise Exception(f"Error with trying to save file at: {save_path}")

    def saveFile(self, data, save_path) -> None:
        try:
            with open(save_path, "w+") as fp:
                fp.write(data)
        except Exception:
            raise Exception(f"Error with trying to save file at: {save_path}")