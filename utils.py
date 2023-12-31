from custom_types import blob_type
from typing import Any, List
import pickle
from timebudget import timebudget
import os
from enum import Enum
from prettytable import PrettyTable


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


def blob_report(blob, kind: str):
    total = 0
    for data in blob:
        total = total + len(data)
    print(f"{kind} : {total} ratings")


def check_and_create(data: Any, filename: Filename, directory: str):
    if not os.path.isfile(f"{directory}{filename}"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        save(data=data, filename=filename, directory=directory)


@timebudget
def save(data: Any, filename: Filename, directory: str):
    print(f"Saving data to {directory}{filename}")
    with open(f"{directory}{filename}", 'wb') as file:
        pickle.dump(data, file)


@timebudget
def load(filename: Filename, directory: str):
    print(f"Loading data from {directory}{filename.value}")
    with open(f"{directory}{filename.value}", 'rb') as file:
        data = pickle.load(file)
    return data


def printTable(data):
    table = PrettyTable()
    table.field_names = ['N', 'ID', 'Name', 'Rating']
    for i, row in enumerate(data):
        table.add_row([f'{i + 1}', row[0], row[1], row[2]])

    print(table)

def divide_chunks(data: List[Any], n: int):
    k, m = divmod(len(data), n)
    return [data[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n)]
