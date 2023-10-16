from custom_types import blob_type
from typing import Any
import joblib


def get_empty_blob(length: int) -> blob_type:
    blob: blob_type = []
    for i in range(length):
        blob.append([])
    return blob


def save(data: Any, path: str):
    joblib.dump(data, filename=path)


def load(path):
    return joblib.load(path)
