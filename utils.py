from custom_types import blob_type
from typing import Any, List
import joblib
import os
from enum import Enum


class Filename(Enum):
    u_indexes = "user_indexes.joblib"
    u_train = "user_training.joblib"
    u_test = "user_testing.joblib"
    u_all = "user_data_blob.joblib"

    i_indexes = "item_indexes.joblib"
    i_train = "item_training.joblib"
    i_test = "item_testing.joblib"
    i_all = "item_data_blob.joblib"


def get_empty_blob(length: int) -> blob_type:
    blob: blob_type = []
    for i in range(length):
        blob.append([])
    return blob


def check_and_create(data: Any, filename: Filename, directory: str):
    print(f"Writing {filename} to {directory}")
    if not os.path.isfile(f"{directory}{filename}"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        save(data=data, filename=filename, directory=directory)


def save(data: Any, filename: Filename, directory: str):
    joblib.dump(data, filename=f"{directory}{filename}")


def load(path):
    return joblib.load(path)


def divide_chunks(data: List[Any], n: int):
    for i in range(0, len(data), n):
        yield data[i:i + n]
