from custom_types import blob_type
from typing import Any, List
import pickle
from timebudget import timebudget
import os
from enum import Enum


class Filename(Enum):
    u_indexes = "user_indexes.pkl"
    u_train = "user_training.pkl"
    u_test = "user_testing.pkl"
    u_all = "user_data_blob.pkl"
    u_vec = "user_vector.pkl"
    u_b = "user_biases.pkl"

    i_indexes = "item_indexes.pkl"
    i_train = "item_training.pkl"
    i_test = "item_testing.pkl"
    i_all = "item_data_blob.pkl"
    i_vec = "item_vector.pkl"
    i_b = "item_biases.pkl"

    f_indexes = "feature_indexes.pkl"
    f_n_ind = "feature_name_indexes.pkl"
    f_ind_n = "feature_index_names.pkl"
    f_vec = "feature_vector.pkl"
    f_items = "feature_items_data_blob.pkl"
    i_features = "item_features_data_blob.pkl"



def get_empty_blob(length: int):
    blob = []
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
    with open(f"{directory}{filename}", 'wb') as file:
        pickle.dump(data, file)


@timebudget
def load(filename: Filename, directory: str):
    with open(f"{directory}{filename.value}", 'rb') as file:
        data = pickle.load(file)
    return data


def divide_chunks(data: List[Any], n: int):
    k, m = divmod(len(data), n)
    return [data[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n)]
