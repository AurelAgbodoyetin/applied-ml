import os.path
from typing import List
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from datasets import Dataset, datasets
from utils import check_and_create, Filename, load, get_empty_blob, blob_report
from timebudget import timebudget
from collections import Counter
from custom_types import blob_type, dict_type, reverse_dict_type, feature_blob_type

np.random.seed(seed=42)


class DataProvider:
    def __init__(self, dataset: Dataset, split_ratio: float = .2):

        self.dataset: Dataset = dataset
        self.dir: str = f"dumps/{self.dataset.name}/"
        self.fig_dir: str = f"figures/{dataset.name}/"
        self.split_ratio: float = split_ratio
        self.ratings_file_lines: List[str] = []

        self.item_indexes: dict_type = dict()
        self.user_indexes: dict_type = dict()
        self.feature_indexes: dict_type = dict()
        self.feature_index_name: reverse_dict_type = dict()
        self.feature_name_index: dict_type = dict()

        self.ratings_file_lines = self.read_file(path=self.dataset.path, shuffle=True)
        self.index_data_file(lines=self.ratings_file_lines)

        self.item_data_blob: blob_type = get_empty_blob(len(self.item_indexes))
        self.item_training_set: blob_type = get_empty_blob(len(self.item_indexes))
        self.item_testing_set: blob_type = get_empty_blob(len(self.item_indexes))

        self.user_data_blob: blob_type = get_empty_blob(len(self.user_indexes))
        self.user_training_set: blob_type = get_empty_blob(len(self.user_indexes))
        self.user_testing_set: blob_type = get_empty_blob(len(self.user_indexes))

        self.feature_items_data_blob: feature_blob_type = get_empty_blob(len(self.feature_indexes))
        self.item_features_data_blob: feature_blob_type = get_empty_blob(len(self.item_indexes))

    def read_file(self, path: str, shuffle: bool):
        file = open(path, "r", encoding="ISO-8859-1")
        lines = file.readlines()
        file.close()
        lines = lines[self.dataset.skip_rows:]
        if shuffle:
            np.random.shuffle(lines)
        return lines

    def index_line(self, line: str, is_feature: bool = False) -> None:
        if is_feature:
            content: List[str] = line.split("|")
            fid: str = content[0]
            name: str = content[1].rstrip()

            if fid not in self.feature_indexes:
                self.feature_index_name[len(self.feature_indexes)] = name
                self.feature_name_index[name] = len(self.feature_indexes)
                self.feature_indexes[fid] = len(self.feature_indexes)
        else:
            content: List[str] = line.split(self.dataset.sep)
            uid: str = content[0]
            iid: str = content[1]

            if uid not in self.user_indexes:
                self.user_indexes[uid] = len(self.user_indexes)

            if iid not in self.item_indexes:
                self.item_indexes[iid] = len(self.item_indexes)

    @timebudget
    def index_data_file(self, lines: List[str]) -> None:
        user_indexes = os.path.isfile(f"{self.dir}{Filename.u_indexes.value}")
        item_indexes = os.path.isfile(f"{self.dir}{Filename.i_indexes.value}")
        feature_indexes = os.path.isfile(f"{self.dir}{Filename.f_indexes.value}")
        if user_indexes and item_indexes and feature_indexes:
            self.item_indexes = load(filename=Filename.i_indexes, directory=self.dir)
            self.user_indexes = load(filename=Filename.u_indexes, directory=self.dir)
            self.feature_indexes = load(filename=Filename.f_indexes, directory=self.dir)
            self.feature_index_name = load(filename=Filename.f_ind_n, directory=self.dir)
            self.feature_name_index = load(filename=Filename.f_n_ind, directory=self.dir)
        else:
            for i in tqdm.trange(len(lines), ascii=True):
                self.index_line(lines[i])
            check_and_create(data=self.user_indexes, filename=Filename.u_indexes.value, directory=self.dir)
            check_and_create(data=self.item_indexes, filename=Filename.i_indexes.value, directory=self.dir)

            with open("data/items.genre") as file:
                features = file.readlines()
                for i in tqdm.trange(len(features), ascii=True):
                    self.index_line(features[i], is_feature=True)

                check_and_create(data=self.feature_indexes, filename=Filename.f_indexes.value, directory=self.dir)
                check_and_create(data=self.feature_name_index, filename=Filename.f_n_ind.value, directory=self.dir)
                check_and_create(data=self.feature_index_name, filename=Filename.f_ind_n.value, directory=self.dir)

    @timebudget
    def extract_set_ds(self, lines: List[str]) -> None:
        print(f"Started extraction whole file DS")
        for i in tqdm.trange(len(lines), ascii=True):
            content = lines[i].split(self.dataset.sep)
            uid: str = content[0]
            iid: str = content[1]
            rating = float(content[2])
            u_index = self.user_indexes[uid]
            i_index = self.item_indexes[iid]

            self.user_data_blob[u_index].append((i_index, rating))
            self.item_data_blob[i_index].append((u_index, rating))

    @timebudget
    def extract_train_test_sets(self) -> None:
        print(f"Started Train - Test extraction")
        for u_index in tqdm.trange(len(self.user_data_blob), ascii=True):
            user_data = self.user_data_blob[u_index]
            test_size = int(np.round(len(user_data) * self.split_ratio))
            train = user_data[:-test_size]
            test = user_data[-test_size:]
            self.user_training_set[u_index] = train
            self.user_testing_set[u_index] = test

            for data in train:
                i_index = data[0]
                rating = data[1]
                self.item_training_set[i_index].append((u_index, rating))

            for data in test:
                i_index = data[0]
                rating = data[1]
                self.item_testing_set[i_index].append((u_index, rating))

    def get_feature_blobs(self):
        skipped = 0
        count = 0
        items_file_lines = self.read_file(path=self.dataset.items_path, shuffle=False)
        for i in tqdm.trange(len(items_file_lines), ascii=True):
            content = items_file_lines[i].split(self.dataset.items_sep)
            iid: str = content[0]
            if iid not in self.item_indexes:
                print(f"{iid} found but not indexed, skipping ...")
                skipped = skipped + 1
                continue

            item_index: int = self.item_indexes[iid]
            features_list: List[str] = content[-1].split(self.dataset.features_sep)
            for feature in features_list:
                count = count + 1
                feature_index = self.feature_name_index[feature.rstrip()]
                self.feature_items_data_blob[feature_index].append(item_index)
                self.item_features_data_blob[item_index].append(feature_index)
        print(f"Counted {count} items")
        print(f"Skipped {skipped} items")

    def plot_power_law(self, save_figure: bool = True) -> None:
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

        if save_figure:
            plt.savefig(f'{self.fig_dir}power_law.pdf')

        plt.show()

    def plot_ratings_distribution(self, save_figure: bool = True):
        all_ratings = [rating[1] for user_ratings in self.user_data_blob for rating in user_ratings]
        counts = Counter(all_ratings)
        plt.bar(counts.keys(), counts.values(), width=0.4)
        plt.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.5)
        plt.xlabel("Rating")
        plt.ylabel("Frequency of Ratings")
        plt.title("Ratings distribution")

        if save_figure:
            plt.savefig(f'{self.fig_dir}ratings_distribution.pdf')

        plt.show()

    def plot_item_count_by_genre(self, save_figure: bool = True):
        features = [feature for feature in self.feature_name_index.keys()]
        counts = [len(self.feature_items_data_blob[i]) for i in self.feature_name_index.values()]
        plt.figure(figsize=(12, 8))
        plt.barh(features, counts)
        plt.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.5)
        plt.xlabel("Items count")
        # plt.ylabel("Count")
        plt.title("Items count by feature")

        if save_figure:
            plt.savefig(f'{self.fig_dir}item_count_by_genre.pdf')

        plt.show()

    def extract_all_file_ds(self):
        self.extract_set_ds(self.ratings_file_lines)
        check_and_create(data=self.user_data_blob, filename=Filename.u_all.value, directory=self.dir)
        check_and_create(data=self.item_data_blob, filename=Filename.i_all.value, directory=self.dir)

        self.get_feature_blobs()
        check_and_create(data=self.feature_items_data_blob, filename=Filename.f_items.value, directory=self.dir)
        check_and_create(data=self.item_features_data_blob, filename=Filename.i_features.value, directory=self.dir)

    def extract_train_test_sets_ds(self):
        self.extract_train_test_sets()
        check_and_create(data=self.user_training_set, filename=Filename.u_train.value, directory=self.dir)
        check_and_create(data=self.item_training_set, filename=Filename.i_train.value, directory=self.dir)
        check_and_create(data=self.user_testing_set, filename=Filename.u_test.value, directory=self.dir)
        check_and_create(data=self.item_testing_set, filename=Filename.i_test.value, directory=self.dir)

    def blobs_report(self):
        blob_report(self.user_data_blob, kind="All")
        blob_report(self.user_training_set, kind="Training")
        blob_report(self.user_testing_set, kind="Testing")
        blob_report(self.feature_items_data_blob, kind="Features")

@timebudget
def get_sparse_matrices_and_dump(dataset: Dataset):
    data_provider: DataProvider = DataProvider(
        dataset=dataset,
        split_ratio=.2,
    )
    data_provider.extract_all_file_ds()
    data_provider.extract_train_test_sets_ds()
    data_provider.blobs_report()
    data_provider.plot_power_law(save_figure=True)
    data_provider.plot_ratings_distribution(save_figure=True)
    data_provider.plot_item_count_by_genre(save_figure=True)
    return data_provider

if __name__ == "__main__":
    provider = get_sparse_matrices_and_dump(dataset=datasets['100k'])
    # provider = get_sparse_matrices_and_dump(dataset=datasets['10m'])
