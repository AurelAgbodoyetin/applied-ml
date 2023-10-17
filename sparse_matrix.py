from typing import List
import numpy as np
from matplotlib import pyplot as plt
from timebudget import timebudget
from joblib import Parallel, delayed
from utils import Filename, divide_chunks, load

from custom_types import blob_type, dict_type, np_type

np.random.seed(seed=42)

class SparseMatrix:
    @timebudget
    def __init__(self, dataset_name: str, n_iter: int = 20, dims: int = 3, parallel: bool = False, tau: float = .01,
                 lambd: float = .01, gamma: float = .01, mu: float = .0, save_figures=False, n_jobs: int = 2):

        self.dir: str = f"joblib_dumps/{dataset_name}/"
        self.fig_dir: str = f"figures/{dataset_name}/"
        self.user_testing_set: blob_type = []
        self.user_training_set: blob_type = []
        # self.user_data_blob: blob_type = []
        self.user_indexes: dict_type = {}
        self.item_testing_set: blob_type = []
        # self.item_data_blob: blob_type = []
        self.item_training_set: blob_type = []
        self.item_indexes: dict_type = {}
        self.save_figures: bool = save_figures
        self.parallel: bool = parallel

        # Initializing model hyper parameters
        self.epochs: int = n_iter
        self.latent_dims: int = dims
        self.tau_: float = tau
        self.lambda_: float = lambd
        self.gamma_: float = gamma
        self.mu: float = mu
        self.sigma: float = np.sqrt(5 / np.sqrt(dims))
        self.n_jobs: int = n_jobs
        self.history = {"training_losses": [], "training_rmse": [], "testing_rmse": []}

        self.load_data()

        # Initializing model parameters
        self.user_biases = np.zeros((len(self.user_indexes)))
        self.item_biases = np.zeros((len(self.item_indexes)))
        self.user_vector = np.random.normal(self.mu, self.sigma, size=(len(self.user_indexes), self.latent_dims))
        self.item_vector = np.random.normal(self.mu, self.sigma, size=(len(self.item_indexes), self.latent_dims))

    def load_data(self):
        self.item_indexes: dict_type = load(filename=Filename.i_indexes, directory=self.dir)
        print(f"Items : {len(self.item_indexes)}")
        # self.item_data_blob: blob_type = load(filename=Filename.i_all, directory=self.dir)
        self.item_training_set: blob_type = load(filename=Filename.i_train, directory=self.dir)
        self.item_testing_set: blob_type = load(filename=Filename.i_test, directory=self.dir)

        self.user_indexes: dict_type = load(filename=Filename.u_indexes, directory=self.dir)
        print(f"Users : {len(self.user_indexes)}")
        # self.user_data_blob: blob_type = load(filename=Filename.u_all, directory=self.dir)
        self.user_training_set: blob_type = load(filename=Filename.u_train, directory=self.dir)
        self.user_testing_set: blob_type = load(filename=Filename.u_test, directory=self.dir)

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

    def update_user_biases(self, users: List[int]) -> np_type:
        result: np_type = np.zeros((len(users)))
        for index, user_index in enumerate(users):
            bias: float = 0
            item_counter: int = 0
            for (item_index, rating) in self.user_training_set[user_index]:
                bias += self.lambda_ * (rating - self.item_biases[item_index] - np.dot(self.user_vector[user_index],
                                                                                       self.item_vector[item_index]))
                item_counter += 1

            result[index] = bias / (self.lambda_ * item_counter + self.gamma_)
        return result

    def update_user_vectors(self, users: List[int]) -> np_type:
        result: np_type = np.zeros((len(users), 3))
        for index, user_index in enumerate(users):
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
            result[index] = np.linalg.inv(L.T) @ np.linalg.inv(L) @ b
        return result

    def update_item_biases(self, items: List[int]) -> np_type:
        result: np_type = np.zeros((len(items)))
        for index, item_index in enumerate(items):
            bias: float = 0.0
            user_counter: int = 0
            for (user_index, rating) in self.item_training_set[item_index]:
                bias += self.lambda_ * (rating - self.user_biases[user_index] - np.dot(self.user_vector[user_index],
                                                                                       self.item_vector[item_index]))
                user_counter += 1

            result[index] = bias / (self.lambda_ * user_counter + self.gamma_)
        return result

    def update_item_vectors(self, items: List[int]) -> np_type:
        result: np_type = np.zeros((len(items), self.latent_dims))
        for index, item_index in enumerate(items):
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
            result[index] = np.linalg.inv(L.T) @ np.linalg.inv(L) @ b
        return result

    @timebudget
    def perform_als(self, parallel=None, dims=None, tau=None, lambd=None, gamma=None, epochs=None) -> None:
        if dims is not None:
            self.latent_dims = dims
            self.user_vector = np.random.normal(self.mu, self.sigma, size=(len(self.user_indexes), self.latent_dims))
            self.item_vector = np.random.normal(self.mu, self.sigma, size=(len(self.item_indexes), self.latent_dims))

        self.tau_ = self.tau_ if tau is None else tau
        self.epochs = self.epochs if epochs is None else epochs
        self.parallel = self.parallel if parallel is None else parallel
        self.lambda_ = self.lambda_ if lambd is None else lambd
        self.gamma_ = self.gamma_ if gamma is None else gamma
        self.history = {"training_losses": [], "training_rmse": [], "testing_rmse": []}

        # Getting data ready
        number_of_users: int = len(self.user_indexes)
        number_of_items: int = len(self.item_indexes)

        for epoch in range(self.epochs):
            if self.parallel:
                user_chunks: List[int] = list(divide_chunks(list(range(number_of_users)), self.n_jobs))
                item_chunks: List[int] = list(divide_chunks(list(range(number_of_items)), self.n_jobs))

                # Updating user biases
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.update_user_biases)(user_chunks[i]) for i in range(self.n_jobs))
                self.user_biases = np.concatenate(results)
                # Updating item biases
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.update_item_biases)(item_chunks[i]) for i in range(self.n_jobs))
                self.item_biases = np.concatenate(results)
                # Updating user vector
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.update_user_vectors)(user_chunks[i]) for i in range(self.n_jobs))
                self.user_vector = np.concatenate(results, axis=0)
                # Updating item vector
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.update_item_vectors)(item_chunks[i]) for i in range(self.n_jobs))
                self.item_vector = np.concatenate(results, axis=0)

            else:
                self.user_biases = self.update_user_biases(list(range(number_of_users)))
                self.item_biases = self.update_item_biases(list(range(number_of_items)))
                self.user_vector = self.update_user_vectors(list(range(number_of_users)))
                self.item_vector = self.update_item_vectors(list(range(number_of_items)))

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
                f'{self.fig_dir}/losses_l_{self.lambda_}_g_{self.gamma_}_t_{self.tau_}_K_{self.latent_dims}.pdf')

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
            plt.savefig(f'{self.fig_dir}/costs_l_{self.lambda_}_g_{self.gamma_}_t_{self.tau_}_K_{self.latent_dims}.pdf')

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

# chunks: List[str] = list(divide_chunks(data_to_use, self.n_threads))
# threads: List[threading.Thread] = [
#     threading.Thread(
#         target=self.get_data_from_line,
#         args=(chunks, is_test, is_whole)
#     ) for _ in range(self.n_threads)
# ]
# for thread in threads:
#     thread.start()

# with mp.ThreadPool(1) as pool:
# pool.starmap(self.get_data_from_line, zip(data_to_use, it.repeat(is_test), it.repeat(is_whole)))
