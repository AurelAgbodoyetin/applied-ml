from typing import List
import itertools as it
import multiprocessing.pool as mp
import numpy as np
from matplotlib import pyplot as plt
from utils import save
from timebudget import timebudget

from custom_types import blob_type, dict_type
from utils import get_empty_blob

np.random.seed(seed=42)
figures_path = "figures"


class DataProvider:
    @timebudget
    def __init__(self, data_path: str, sep: str = ",", skip_rows: int = 0, split_ratio: float = .2, dump: bool = False,
                 parallel: bool = False, save_figures=False, show_power_law=False):

        self.data_path: str = data_path
        self.sep: str = sep
        self.dump = dump
        self.skip_rows: int = skip_rows
        self.save_figures: bool = save_figures
        self.parallel: bool = parallel
        self.split_ratio: float = split_ratio
        self.file_lines: List[str] = []

        self.item_indexes: dict_type = dict()
        self.user_indexes: dict_type = dict()

        self.read_file()
        self.index_data_file(lines=self.file_lines)

        self.item_data_blob: blob_type = get_empty_blob(len(self.item_indexes))
        self.item_training_set: blob_type = get_empty_blob(len(self.item_indexes))
        self.item_testing_set: blob_type = get_empty_blob(len(self.item_indexes))

        self.user_data_blob: blob_type = get_empty_blob(len(self.user_indexes))
        self.user_training_set: blob_type = get_empty_blob(len(self.user_indexes))
        self.user_testing_set: blob_type = get_empty_blob(len(self.user_indexes))

        if show_power_law:
            self.plot_all_file_ds()

    def read_file(self):
        file = open(self.data_path, "r", encoding="ISO-8859-1")
        lines = file.readlines()
        file.close()
        lines = lines[self.skip_rows:]
        np.random.shuffle(lines)
        self.file_lines = lines

    def index_line(self, line: str) -> None:
        content: List[str] = line.split(self.sep)
        uid: str = content[0]
        iid: str = content[1]

        if uid not in self.user_indexes:
            self.user_indexes[uid] = len(self.user_indexes)

        if iid not in self.item_indexes:
            self.item_indexes[iid] = len(self.item_indexes)

    @timebudget
    def index_data_file(self, lines: List[str]) -> None:
        print("Started indexing")
        for line in lines:
            self.index_line(line)

        if self.dump:
            save(data=self.user_indexes, path="joblib_dumps/user_indexes.joblib")
            save(data=self.item_indexes, path="joblib_dumps/item_indexes.joblib")

    @timebudget
    def extract_set_ds(self, lines: List[str], is_test: bool = False, is_whole: bool = False) -> None:
        print(f"Started extraction whole: {is_whole} test: {is_test}")
        test_split_size = int(np.round(len(lines) * self.split_ratio))
        data_to_use = lines if is_whole else lines[-test_split_size:] if is_test else lines[:-test_split_size]

        if self.parallel:
            with mp.ThreadPool(1) as pool:
                pool.starmap(self.get_data_from_line, zip(data_to_use, it.repeat(is_test), it.repeat(is_whole)))
        else:
            for line in data_to_use:
                self.get_data_from_line(line, is_test, is_whole)

        if is_whole:
            print("Whole sparse matrix extraction complete")
        else:
            if is_test:
                print("Test sparse matrix extraction complete")
            else:
                print("Train sparse matrix extraction complete")

    def get_data_from_line(self, line: str, is_test: bool, is_whole: bool) -> None:
        content = line.split(self.sep)
        uid: str = content[0]
        iid: str = content[1]
        rating = float(content[2])

        u_index = self.user_indexes[uid]
        i_index = self.item_indexes[iid]

        if is_whole:
            self.user_data_blob[u_index].append((i_index, rating))
            self.item_data_blob[i_index].append((u_index, rating))
        else:
            if is_test:
                self.user_testing_set[u_index].append((self.item_indexes[iid], rating))
                self.item_testing_set[i_index].append((self.user_indexes[uid], rating))
            else:
                self.user_training_set[u_index].append((self.item_indexes[iid], rating))
                self.item_training_set[i_index].append((self.user_indexes[uid], rating))

    def plot_all_file_ds(self) -> None:
        fig = plt.figure(figsize=(7, 5))
        u_degrees: List[int] = [len(ratings) for ratings in self.user_data_blob]
        u_frequencies: List[int] = [u_degrees.count(degree) for degree in u_degrees]
        plt.scatter(u_degrees, u_frequencies, c="b", marker="*", s=1)

        i_degrees: List[int] = [len(ratings) for ratings in self.item_data_blob]
        i_frequencies: List[int] = [i_degrees.count(degree) for degree in i_degrees]
        plt.scatter(i_degrees, i_frequencies, c="r", marker="s", s=1)

        plt.xscale('log')
        plt.yscale('log')
        plt.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.5)
        plt.xlabel("Rating Scale")
        plt.ylabel("Frequency of Ratings")
        plt.title("Frequency of Ratings")
        plt.legend(['Users', 'Items'])

        if self.save_figures:
            plt.savefig(f'{figures_path}/power_law.pdf')
        plt.show()

    def extract_all_file_ds(self):
        self.extract_set_ds(self.file_lines, is_whole=True)
        if self.dump:
            save(data=self.user_data_blob, path="joblib_dumps/user_data_blob.joblib")
            save(data=self.item_data_blob, path="joblib_dumps/item_data_blob.joblib")

    def extract_train_set_ds(self):
        self.extract_set_ds(self.file_lines, is_whole=False, is_test=False)
        if self.dump:
            save(data=self.user_training_set, path="joblib_dumps/user_training_set.joblib")
            save(data=self.item_training_set, path="joblib_dumps/item_training_set.joblib")

    def extract_test_set_ds(self):
        self.extract_set_ds(self.file_lines, is_test=True)
        if self.dump:
            save(data=self.user_testing_set, path="joblib_dumps/user_testing_set.joblib")
            save(data=self.item_testing_set, path="joblib_dumps/item_testing_set.joblib")
