from custom_types import blob_type
from joblib import load, dump


def get_empty_blob(length: int) -> blob_type:
    blob: blob_type = []
    for i in range(length):
        blob.append([])

    return blob


class StorageManager:
    def __init__(self, path: str, filename: str):
        self.path = path
        self.filename = filename

    def save(self, data):
        dump(data, f'{self.path}{self.filename}.joblib')

    def load(self):
        return load(f'{self.path}{self.filename}.joblib')
