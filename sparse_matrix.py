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


class SparseMatrix:
    #@timebudget
    @timeit
    def __init__(self, data_path: str, sep: str = ",", skip_rows: int = 0, split_ratio: float = .2, n_iter: int = 20,
                 dims: int = 3, parallel_extraction: bool = False, parallel_als: bool = False, tau: float = .01,
                 lambd: float = .01, gamma: float = .01, mu: float = .0, save_figures=False, show_power_law=False,
                 parallel_indexing: bool = False):
        self.data_path: str = data_path
        self.sep: str = sep
        self.skip_rows: int = skip_rows
        self.save_figures: bool = save_figures
        self.user_indexes: dict_type = dict()
        self.item_indexes: dict_type = dict()
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

        self.item_data_blob: blob_type = get_empty_blob(len(self.item_indexes))
        self.item_training_set: blob_type = get_empty_blob(len(self.item_indexes))
        self.item_testing_set: blob_type = get_empty_blob(len(self.item_indexes))

        self.user_data_blob: blob_type = get_empty_blob(len(self.user_indexes))
        self.user_training_set: blob_type = get_empty_blob(len(self.user_indexes))
        self.user_testing_set: blob_type = get_empty_blob(len(self.user_indexes))

        self.extract_set_ds(file_lines, is_whole=True)
        self.extract_set_ds(file_lines, is_test=True)
        self.extract_set_ds(file_lines, is_test=False)

        # Initializing model parameters
        self.user_biases = np.zeros((len(self.user_indexes)))
        self.item_biases = np.zeros((len(self.item_indexes)))
        self.user_vector = np.random.normal(self.mu, self.sigma, size=(len(self.user_indexes), self.latent_dims))
        self.item_vector = np.random.normal(self.mu, self.sigma, size=(len(self.item_indexes), self.latent_dims))

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

    def log_likelihood(self) -> float:
        s: float = 0
        for m in range(len(self.user_biases)):
            for n, r in self.user_training_set[m]:
                error = np.dot(self.user_vector[m], self.item_vector[n]) + self.user_biases[m] + self.item_biases[n]
                error = (r - error) ** 2
                s = s + error

        loss = - self.lambda_ * s / 2

        user_vector_term: float = 0
        for m in range(len(self.user_biases)):
            user_vector_term = user_vector_term + np.dot(self.user_vector[m], self.user_vector[m])  # .T

        user_vector_term: float = - user_vector_term * self.tau_ / 2
        user_biases_regularizer = - np.dot(self.user_biases, self.user_biases) * self.gamma_ / 2

        item_vector_term: float = 0
        for n in range(len(self.item_biases)):
            item_vector_term = item_vector_term + np.dot(self.item_vector[n], self.item_vector[n])  # .T

        item_vector_term: float = - item_vector_term * self.tau_ / 2
        item_biases_regularizer = - np.dot(self.item_biases, self.item_biases) * self.gamma_ / 2

        loss = loss + user_vector_term + item_vector_term + user_biases_regularizer + item_biases_regularizer
        return loss

    def update_user_bias(self, user_index: int) -> float:
        bias: float = 0
        item_counter: int = 0
        for (item_index, rating) in self.user_training_set[user_index]:
            bias += self.lambda_ * (rating - self.item_biases[item_index] - np.dot(self.user_vector[user_index],
                                                                                   self.item_vector[item_index]))
            item_counter += 1
        return bias / (self.lambda_ * item_counter + self.gamma_)

    def update_user_vector(self, user_index) -> np_type:
        tau_matrix: np_type = self.tau_ * np.eye(self.latent_dims)
        s: np_type = np.zeros((self.latent_dims, self.latent_dims))
        b: np_type = np.zeros(self.latent_dims)
        for item_index, rating in self.user_training_set[user_index]:
            s = s + np.outer(self.item_vector[item_index], self.item_vector[item_index])
            b = b + self.item_vector[item_index, :] * (
                        rating - self.user_biases[user_index] - self.item_biases[item_index])

        A: np_type = self.lambda_ * s + tau_matrix
        b: np_type = self.lambda_ * b
        L: np_type = np.linalg.cholesky(A)
        return np.linalg.inv(L.T) @ np.linalg.inv(L) @ b

    def update_item_bias(self, item_index: int) -> float:
        bias: float = 0.0
        user_counter: int = 0
        for (user_index, rating) in self.item_training_set[item_index]:
            bias += self.lambda_ * (rating - self.user_biases[user_index] - np.dot(self.user_vector[user_index],
                                                                                   self.item_vector[item_index]))
            user_counter += 1
        return bias / (self.lambda_ * user_counter + self.gamma_)

    def update_item_vector(self, item_index: int) -> np_type:
        tau_matrix: np_type = self.tau_ * np.eye(self.latent_dims)
        s: np_type = np.zeros((self.latent_dims, self.latent_dims))
        b: np_type = np.zeros(self.latent_dims)
        for (user_index, rating) in self.item_training_set[item_index]:
            s = s + np.outer(self.user_vector[user_index], self.user_vector[user_index])
            b = b + self.user_vector[user_index] * (
                        rating - self.user_biases[user_index] - self.item_biases[item_index])

        A: np_type = self.lambda_ * s + tau_matrix
        b: np_type = self.lambda_ * b
        L: np_type = np.linalg.cholesky(A)
        return np.linalg.inv(L.T) @ np.linalg.inv(L) @ b

    #@timebudget
    @timeit
    def perform_als(self, is_parallel: bool, dims=None, tau=None, lambd=None, gamma=None) -> None:
        if dims is not None:
            self.latent_dims = dims
            self.user_vector = np.random.normal(self.mu, self.sigma, size=(len(self.user_indexes), self.latent_dims))
            self.item_vector = np.random.normal(self.mu, self.sigma, size=(len(self.item_indexes), self.latent_dims))

        self.tau_ = self.tau_ if tau is None else tau
        self.lambda_ = self.lambda_ if lambd is None else lambd
        self.gamma_ = self.gamma_ if gamma is None else gamma
        self.history = {"training_losses": [], "training_rmse": [], "testing_rmse": []}

        # Getting data ready
        number_of_users: int = len(self.user_indexes)
        number_of_items: int = len(self.item_indexes)

        for epoch in range(self.epochs):
            if is_parallel:
                pool: mp.Pool = mp.Pool(processes=10)
                self.user_biases = pool.map(self.update_user_bias, list(range(number_of_users)), chunksize=10)
                self.item_biases = pool.map(self.update_item_bias, list(range(number_of_items)), chunksize=10)
                self.user_vector = pool.map(self.update_user_vector, list(range(number_of_users)), chunksize=10)
                self.item_vector = pool.map(self.update_item_vector, list(range(number_of_items)), chunksize=10)
                pool.close()
                pool.join()
            else:
                for m in range(number_of_users):
                    self.user_biases[m] = self.update_user_bias(m)

                for n in range(number_of_items):
                    self.item_biases[n] = self.update_item_bias(n)

                for m in range(number_of_users):
                    self.user_vector[m] = self.update_user_vector(m)

                for n in range(number_of_items):
                    self.item_vector[n] = self.update_item_vector(n)

            loss = self.log_likelihood()
            self.history["training_losses"].append(loss)

            training_cost = self.rmse(is_test=False)
            self.history["training_rmse"].append(training_cost)

            testing_cost = self.rmse(is_test=True)
            self.history["testing_rmse"].append(testing_cost)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"K = {self.latent_dims} --> Iteration {epoch + 1} : Loss = {-loss:.4f} , "
                      f"Training cost = {training_cost:.4f}, Testing cost = {testing_cost:.4f}")

        self.plot_losses()
        self.plot_rmse()

    def get_parameters_str(self) -> str:
        return '\n'.join((
            r'$K=%d$' % (self.latent_dims,),
            r'$\tau=%.2f$' % (self.tau_,),
            r'$\lambda=%.2f$' % (self.lambda_,),
            r'$\gamma=%.2f$' % (self.gamma_,)
        ))

    def plot_losses(self) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = range(1, len(self.history["training_losses"]) + 1)

        ax.plot(x, [-k for k in self.history["training_losses"]], marker="d")
        ax.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Iterations")
        ax.set_title("Losses during Training")
        ax.set_xticks(x)
        ax.text(0.85, 0.5, self.get_parameters_str(), transform=ax.transAxes, fontsize=11,
                horizontalalignment="left", verticalalignment="center",
                # bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                bbox=dict(boxstyle="round", fc="white", ec="black", pad=0.2),
                )

        if self.save_figures:
            plt.savefig(
                f'{figures_path}/losses_l_{self.lambda_}_g_{self.gamma_}_t_{self.tau_}_K_{self.latent_dims}.pdf')

        plt.show()

    def plot_rmse(self) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = range(1, len(self.history["training_rmse"]) + 1)

        ax.plot(x, self.history["training_rmse"], marker="+")
        ax.plot(x, self.history["testing_rmse"], marker="4")
        ax.legend(["Training RMSE", "Testing RMSE"])
        ax.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.5)
        ax.set_title("Training cost vs Testing cost")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Iterations")
        ax.set_xticks(x)
        ax.text(0.85, 0.5, self.get_parameters_str(), transform=ax.transAxes, fontsize=11,
                horizontalalignment="left", verticalalignment="center",
                # bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                bbox=dict(boxstyle="round", fc="white", ec="black", pad=0.2, alpha=0.5),
                )

        if self.save_figures:
            plt.savefig(f'{figures_path}/costs_l_{self.lambda_}_g_{self.gamma_}_t_{self.tau_}_K_{self.latent_dims}.pdf')

        plt.show()

    def get_predictions(self, is_test) -> (List[float], List[float]):
        user_map = self.user_indexes
        u_data_blob = self.user_testing_set if is_test else self.user_training_set
        number_of_users = len(user_map)
        predictions: List[float] = []
        targets: List[float] = []
        for m in range(number_of_users):
            for n, r in u_data_blob[m]:
                pred = np.dot(self.user_vector[m], self.item_vector[n]) + self.user_biases[m] + self.item_biases[n]
                targets.append(r)
                predictions.append(pred)

        return targets, predictions

    def rmse(self, is_test) -> float:
        targets, predictions = self.get_predictions(is_test)
        mse = np.square(np.subtract(targets, predictions)).mean()
        return np.sqrt(mse)
