from typing import List
import itertools as it
import multiprocessing as mp
from timeit import timeit
import numpy as np
from matplotlib import pyplot as plt
#from timebudget import timebudget

from custom_types import blob_type, dict_type, np_type
from utils import get_empty_blob

np.random.seed(seed=42)
figures_path = "figures"


class DataProvider:
    #@timebudget
    @timeit
    def __init__(self, data_path: str, sep: str = ",", skip_rows: int = 0, split_ratio: float = .2, 
                parallel_extraction: bool = False, parallel_als: bool = False, save_figures=False, 
                show_power_law=False, parallel_indexing: bool = False):

        self.data_path: str = data_path
        self.sep: str = sep
        self.skip_rows: int = skip_rows
        self.save_figures: bool = save_figures
        self.parallel_extraction: bool = parallel_extraction
        self.parallel_als: bool = parallel_als
        self.parallel_indexing: bool = parallel_indexing

        # Initializing model hyper parameters
        self.epochs: int = n_iter
        self.split_ratio: float = split_ratio
        self.latent_dims: int = dims
        self.tau_: float = tau
        self.lambda_: float = lambd
        self.gamma_: float = gamma
        self.mu: float = mu
        self.sigma: float = np.sqrt(5 / np.sqrt(dims))
        self.history = {"training_losses": [], "training_rmse": [], "testing_rmse": []}

        file_lines: List[str] = self.read_file()
        self.index_data_file(lines=file_lines)
        
        self.item_indexes: dict_type = dict()
        self.item_data_blob: blob_type = get_empty_blob(len(self.item_indexes))
        self.item_training_set: blob_type = get_empty_blob(len(self.item_indexes))
        self.item_testing_set: blob_type = get_empty_blob(len(self.item_indexes))

        self.user_indexes: dict_type = dict()
        self.user_data_blob: blob_type = get_empty_blob(len(self.user_indexes))
        self.user_training_set: blob_type = get_empty_blob(len(self.user_indexes))
        self.user_testing_set: blob_type = get_empty_blob(len(self.user_indexes))

        self.extract_set_ds(file_lines, is_whole=True)
        self.extract_set_ds(file_lines, is_test=True)
        self.extract_set_ds(file_lines, is_test=False)

        if show_power_law:
            self.plot_all_file_ds()

    def read_file(self) -> List[str]:
        file = open(self.data_path, "r", encoding="ISO-8859-1")
        lines = file.readlines()
        file.close()
        lines = lines[self.skip_rows:]
        np.random.shuffle(lines)
        return lines

    def index_line(self, line: str) -> None:
        content = line.split(self.sep)
        uid: str = content[0]
        iid: str = content[1]

        if uid not in self.user_indexes:
            self.user_indexes[uid] = len(self.user_indexes)

        if iid not in self.item_indexes:
            self.item_indexes[iid] = len(self.item_indexes)

    #@timebudget
    @timeit
    def index_data_file(self, lines: List[str]) -> None:
        print("Started indexing")
        if self.parallel_indexing:
            pool = mp.Pool(processes=mp.cpu_count())
            pool.map(self.index_line, lines)
            pool.close()
            pool.join()
        else:
            for line in lines:
                self.index_line(line)

    #@timebudget
    @timeit
    def extract_set_ds(self, lines: List[str], is_test: bool = False, is_whole: bool = False) -> None:
        print(f"Started extraction whole: {is_whole} test: {is_test}")
        test_split_size = int(np.round(len(lines) * self.split_ratio))
        data_to_use = lines if is_whole else lines[-test_split_size:] if is_test else lines[:-test_split_size]

        if self.parallel_extraction:
            pool = mp.Pool(processes=mp.cpu_count())
            pool.starmap(self.get_data_from_line, zip(data_to_use, it.repeat(is_test), it.repeat(is_whole)))
            pool.close()
            pool.join()
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
            # files.download('power_law.pdf')
        plt.show()

    