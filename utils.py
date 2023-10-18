from custom_types import blob_type
from typing import Any, List
import joblib
from timebudget import timebudget
import os
from enum import Enum
import numpy as np
from multiprocessing import shared_memory


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

def create_shared_memory_nparray(data, name:str, dtype:np.dtype, shape:tuple):
    d_size = np.dtype(dtype).itemsize * np.prod(shape)
    shm = shared_memory.SharedMemory(create=True, size=d_size, name=name)
    dst = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    dst[:] = data[:]
    print(f'NP SIZE: {(dst.nbytes / 1024) / 1024}')
    return shm, dst


def release_shared(name):
    shm = shared_memory.SharedMemory(name=name)
    shm.close()
    shm.unlink()
