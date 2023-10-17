from custom_types import blob_type
from typing import Any, List
import joblib
from timebudget import timebudget
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
    if not os.path.isfile(f"{directory}{filename}"):
        print(f"Writing {filename} to {directory}")
        if not os.path.exists(directory):
            os.makedirs(directory)
        save(data=data, filename=filename, directory=directory)


@timebudget
def save(data: Any, filename: Filename, directory: str):
    joblib.dump(data, filename=f"{directory}{filename}")


@timebudget
def load(filename: Filename, directory: str):
    return joblib.load(f"{directory}{filename.value}")


def divide_chunks(data: List[Any], n: int):
    k, m = divmod(len(data), n)
    return [data[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n)]

# print(divide_chunks(range(30), 3))
