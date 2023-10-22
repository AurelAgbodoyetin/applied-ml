from random import sample
from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt
from timebudget import timebudget
from concurrent.futures import ThreadPoolExecutor
from datasets import Dataset
from utils import Filename, load, check_and_create, get_empty_blob
import tqdm
from custom_types import blob_type, dict_type, np_type, reverse_dict_type, feature_blob_type

np.random.seed(seed=42)


class ALSModel:
    @timebudget
    def __init__(self, dataset: Dataset, biases_only: bool, use_features: bool, n_iter: int = 20, dims: int = 3,
                 parallel: bool = False, tau: float = .01, lambd: float = .01, gamma: float = .01, mu: float = .0,
                 save_figures=False, n_jobs: int = 2):

        self.dumps_dir: str = f"dumps/{dataset.name}/"
        self.fig_dir: str = f"figures/{dataset.name}/"
        self.models_dir: str = f"models/{dataset.name}/"

        self.user_testing_set: blob_type = []
        self.user_training_set: blob_type = []
        self.user_data_blob: blob_type = []
        self.user_indexes: dict_type = {}
        self.reverse_user_indexes: reverse_dict_type = {}

        self.item_testing_set: blob_type = []
        self.item_data_blob: blob_type = []
        self.item_training_set: blob_type = []
        self.item_indexes: dict_type = {}
        self.reverse_item_indexes: reverse_dict_type = {}
        self.feature_indexes: dict_type = {}

        self.feature_index_name: reverse_dict_type = {}
        self.feature_name_index: dict_type = {}
        self.feature_items_data_blob: feature_blob_type = []
        self.item_features_data_blob: feature_blob_type = []

        self.dataset: Dataset = dataset
        self.biases_only = biases_only
        self.use_features = use_features
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

        self.load_dataset()

        # Initializing model parameters
        self.user_biases = np.zeros((len(self.user_indexes)))
        self.item_biases = np.zeros((len(self.item_indexes)))
        self.user_vector = np.random.normal(self.mu, self.sigma, size=(len(self.user_indexes), self.latent_dims))
        self.item_vector = np.random.normal(self.mu, self.sigma, size=(len(self.item_indexes), self.latent_dims))
        # TODO Remove one here
        # self.feature_vector = np.random.normal(self.mu, self.sigma, size=(len(self.item_indexes), self.latent_dims))
        self.feature_vector = np.zeros((len(self.feature_indexes), self.latent_dims))

    def load_dataset(self):
        self.item_indexes = load(filename=Filename.i_indexes, directory=self.dumps_dir)
        self.reverse_item_indexes = {v: k for k, v in self.item_indexes.items()}
        print(f"Items : {len(self.item_indexes)}")
        self.item_data_blob = load(filename=Filename.i_all, directory=self.dumps_dir)
        self.item_training_set = load(filename=Filename.i_train, directory=self.dumps_dir)
        self.item_testing_set = load(filename=Filename.i_test, directory=self.dumps_dir)

        self.user_indexes = load(filename=Filename.u_indexes, directory=self.dumps_dir)
        self.reverse_user_indexes = {v: k for k, v in self.user_indexes.items()}
        print(f"Users : {len(self.user_indexes)}")
        self.user_data_blob = load(filename=Filename.u_all, directory=self.dumps_dir)
        self.user_training_set = load(filename=Filename.u_train, directory=self.dumps_dir)
        self.user_testing_set = load(filename=Filename.u_test, directory=self.dumps_dir)

        self.feature_indexes = load(filename=Filename.f_indexes, directory=self.dumps_dir)
        self.feature_name_index = load(filename=Filename.f_n_ind, directory=self.dumps_dir)
        self.feature_index_name = load(filename=Filename.f_ind_n, directory=self.dumps_dir)
        print(f"Features : {len(self.feature_indexes)}")
        self.feature_items_data_blob = load(filename=Filename.f_items, directory=self.dumps_dir)
        self.item_features_data_blob = load(filename=Filename.i_features, directory=self.dumps_dir)

    def log_likelihood(self) -> float:
        s: float = 0
        for m in range(len(self.user_biases)):
            for n, r in self.user_training_set[m]:
                if self.biases_only:
                    error = self.user_biases[m] + self.item_biases[n]
                else:
                    error = np.dot(self.user_vector[m], self.item_vector[n]) + self.user_biases[m] + self.item_biases[n]
                error = (r - error) ** 2
                s = s + error

        error_term = - self.lambda_ * s / 2

        user_biases_regularizer = - np.dot(self.user_biases, self.user_biases) * self.gamma_ / 2
        item_biases_regularizer = - np.dot(self.item_biases, self.item_biases) * self.gamma_ / 2

        user_vector_term: float = 0
        item_vector_term: float = 0
        if not self.biases_only:
            for m in range(len(self.user_biases)):
                user_vector_term = user_vector_term + np.dot(self.user_vector[m], self.user_vector[m])
            user_vector_term = - user_vector_term * self.tau_ / 2

            for n in range(len(self.item_biases)):
                v_center = np.zeros(self.latent_dims)
                if self.use_features:
                    fl = np.zeros(self.latent_dims)
                    for feature_index in self.item_features_data_blob[n]:
                        fl = fl + self.feature_vector[feature_index]
                    v_center = fl / np.sqrt(len(self.item_features_data_blob[n]))
                item_vector_term = item_vector_term + np.dot(self.item_vector[n] - v_center,
                                                             self.item_vector[n] - v_center)
            item_vector_term = - item_vector_term * self.tau_ / 2

        feature_vector_term: float = 0
        if self.use_features:
            for f in range(len(self.feature_indexes)):
                feature_vector_term = feature_vector_term + np.dot(self.feature_vector[f], self.feature_vector[f])
            feature_vector_term = - feature_vector_term * self.tau_ / 2

        loss = (error_term + user_vector_term + item_vector_term + feature_vector_term + user_biases_regularizer +
                item_biases_regularizer)
        return loss

    def update_users(self, users: List[int]):
        for user_index in users:
            bias: float = 0
            item_counter: int = 0
            for (item_index, rating) in self.user_training_set[user_index]:
                bias += self.lambda_ * (rating - self.item_biases[item_index] - np.dot(self.user_vector[user_index],
                                                                                       self.item_vector[item_index]))
                item_counter += 1

            self.user_biases[user_index] = bias / (self.lambda_ * item_counter + self.gamma_)

            if not self.biases_only:
                tau_matrix: np_type = self.tau_ * np.eye(self.latent_dims)
                s: np_type = np.zeros((self.latent_dims, self.latent_dims))
                b: np_type = np.zeros(self.latent_dims)

                for item_index, rating in self.user_training_set[user_index]:
                    s = s + np.outer(self.item_vector[item_index], self.item_vector[item_index])
                    b = b + self.item_vector[item_index, :] * (
                            rating - self.user_biases[user_index] - self.item_biases[item_index])

                a: np_type = self.lambda_ * s + tau_matrix
                b: np_type = self.lambda_ * b
                l: np_type = np.linalg.cholesky(a)
                self.user_vector[user_index] = np.linalg.inv(l.T) @ np.linalg.inv(l) @ b

    def update_items(self, items: List[int]):
        for item_index in items:
            bias: float = 0.0
            user_counter: int = 0
            for (user_index, rating) in self.item_training_set[item_index]:
                bias += self.lambda_ * (rating - self.user_biases[user_index] - np.dot(self.user_vector[user_index],
                                                                                       self.item_vector[item_index]))
                user_counter += 1

            self.item_biases[item_index] = bias / (self.lambda_ * user_counter + self.gamma_)

            if not self.biases_only:
                tau_matrix: np_type = self.tau_ * np.eye(self.latent_dims)
                s: np_type = np.zeros((self.latent_dims, self.latent_dims))
                b: np_type = np.zeros(self.latent_dims)
                features_term: np_type = np.zeros(self.latent_dims)

                for (user_index, rating) in self.item_training_set[item_index]:
                    s = s + np.outer(self.user_vector[user_index], self.user_vector[user_index])
                    b = b + self.user_vector[user_index] * (
                            rating - self.user_biases[user_index] - self.item_biases[item_index])

                if self.use_features:
                    for feature_index in self.item_features_data_blob[item_index]:
                        features_term = features_term + self.feature_vector[feature_index]

                a: np_type = self.lambda_ * s + tau_matrix
                b = self.lambda_ * b + self.tau_ * features_term
                l: np_type = np.linalg.cholesky(a)
                self.item_vector[item_index] = np.linalg.inv(l.T) @ np.linalg.inv(l) @ b

    def update_features(self, features: List[int]):
        for feature_index in features:
            s_fn = 0
            s_vec = np.zeros(self.latent_dims)
            feature_items = self.feature_items_data_blob[feature_index]
            for item_index in feature_items:
                fn = len(self.item_features_data_blob[item_index])
                s_fn = s_fn + fn
                s_vec = s_vec + self.item_vector[item_index] / np.sqrt(fn)
            self.feature_vector[feature_index] = (1 / (1 + s_fn)) * s_vec

    @timebudget
    def train(self, save_best: bool, plot: bool, parallel=None, dims=None, tau=None, lambd=None, gamma=None,
              epochs=None, biases_only=None) -> None:
        if dims is not None:
            self.latent_dims = dims
            self.user_vector = np.random.normal(self.mu, self.sigma, size=(len(self.user_indexes), self.latent_dims))
            self.item_vector = np.random.normal(self.mu, self.sigma, size=(len(self.item_indexes), self.latent_dims))

        self.biases_only = self.biases_only if biases_only is None else biases_only
        self.tau_ = self.tau_ if tau is None else tau
        self.epochs = self.epochs if epochs is None else epochs
        self.parallel = self.parallel if parallel is None else parallel
        self.lambda_ = self.lambda_ if lambd is None else lambd
        self.gamma_ = self.gamma_ if gamma is None else gamma
        self.history = {"training_losses": [], "training_rmse": [], "testing_rmse": []}

        # Getting data ready
        number_of_users: int = len(self.user_indexes)
        number_of_items: int = len(self.item_indexes)
        number_of_features: int = len(self.feature_indexes)
        for epoch in tqdm.trange(self.epochs, ascii=True):
            if self.parallel:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    executor.submit(self.update_users, list(range(number_of_users)))
                    executor.submit(self.update_items, list(range(number_of_items)))
                    executor.submit(self.update_features, list(range(number_of_features)))
            else:
                self.update_users(list(range(number_of_users)))
                self.update_items(list(range(number_of_items)))
                self.update_features(list(range(number_of_features)))

            loss = self.log_likelihood()
            training_rmse = self.rmse(is_test=False)
            testing_rmse = self.rmse(is_test=True)
            if save_best:
                if epoch == 0 or testing_rmse < self.history["testing_rmse"][-1]:
                    self.save_parameters(epoch + 1, loss, training_rmse, testing_rmse)

            self.history["training_losses"].append(loss)
            self.history["training_rmse"].append(training_rmse)
            self.history["testing_rmse"].append(testing_rmse)

            if epoch + 1 == self.epochs:
                print(f"K = {self.latent_dims} --> Iteration {epoch + 1} : Loss = {-loss:.4f} , "
                      f"Training RMSE = {training_rmse:.4f}, Testing RMSE = {testing_rmse:.4f}")

        if plot:
            self.plot_losses()
            self.plot_rmse()

    def save_parameters(self, iteration, loss, training_rmse, testing_rmse):
        check_and_create(data=self.user_vector, filename=Filename.u_vec.value, directory=self.models_dir)
        check_and_create(data=self.item_vector, filename=Filename.i_vec.value, directory=self.models_dir)
        check_and_create(data=self.feature_vector, filename=Filename.f_vec.value, directory=self.models_dir)
        check_and_create(data=self.user_biases, filename=Filename.u_b.value, directory=self.models_dir)
        check_and_create(data=self.item_biases, filename=Filename.i_b.value, directory=self.models_dir)
        with open(f'{self.models_dir}model.data', 'w') as f:
            f.write(f"biases_only={self.biases_only}\nuse_features={self.use_features}\nK = {self.latent_dims}\n"
                    f"Iteration = {iteration}/{self.epochs}\nLoss = {-loss:.4f}\nTraining RMSE = {training_rmse:.4f}\n"
                    f"Testing RMSE = {testing_rmse:.4f}")

    def load_parameters(self):
        perf = []
        biases_only: bool = False
        use_features: bool = False

        with open(f'{self.models_dir}model.data', 'r') as f:
            for index, line in enumerate(f.readlines()):
                if index == 0:
                    biases_only = bool(line.split("=")[1])
                elif index == 1:
                    use_features = bool(line.split("=")[1])
                else:
                    perf.append(line)

        self.user_biases = load(filename=Filename.u_b, directory=self.models_dir)
        self.item_biases = load(filename=Filename.i_b, directory=self.models_dir)

        if not biases_only:
            self.user_vector = load(filename=Filename.u_vec, directory=self.models_dir)
            self.item_vector = load(filename=Filename.i_vec, directory=self.models_dir)

        if use_features:
            self.feature_vector = load(filename=Filename.f_vec, directory=self.models_dir)

        print("MODEL LOADED")
        for line in perf:
            print(line, end="")
        print()

    def get_parameters_str(self) -> str:
        return '\n'.join((
            r'$K=%d$' % (self.latent_dims,),
            r'$\tau=%.2f$' % (self.tau_,),
            r'$\lambda=%.2f$' % (self.lambda_,),
            r'$\gamma=%.2f$' % (self.gamma_,)
        ))

    def plot_losses(self, save: bool = None) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = range(1, len(self.history["training_losses"]) + 1)
        ax.plot(x, [-k for k in self.history["training_losses"]], marker="d")
        ax.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Iterations")
        ax.set_title("Losses during Training")
        ax.set_xticks(x)
        ax.text(0.85, 0.5, self.get_parameters_str(), transform=ax.transAxes, fontsize=11,
                horizontalalignment="left", verticalalignment="center",
                bbox=dict(boxstyle="round", fc="white", ec="black", pad=0.2),
                )
        if self.save_figures or save:
            plt.savefig(
                f'{self.fig_dir}/losses_l_{self.lambda_}_g_{self.gamma_}_t_{self.tau_}_K_{self.latent_dims}.pdf'
            )

        plt.show()

    def plot_rmse(self, save: bool = None) -> None:
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

        if self.save_figures or save:
            plt.savefig(f'{self.fig_dir}/costs_l_{self.lambda_}_g_{self.gamma_}_t_{self.tau_}_K_{self.latent_dims}.pdf')

        plt.show()

    def predict(self, m: int, n: int, is_rmse=True):
        if is_rmse:
            return np.dot(self.user_vector[m], self.item_vector[n]) + self.user_biases[m] + self.item_biases[n]
        else:
            return np.dot(self.user_vector[m], self.item_vector[n]) + self.user_biases[m]

    def get_predictions(self, is_test) -> (List[float], List[float]):
        user_map = self.user_indexes
        u_data_blob = self.user_testing_set if is_test else self.user_training_set
        number_of_users = len(user_map)
        predictions: List[float] = []
        targets: List[float] = []
        for m in range(number_of_users):
            for n, r in u_data_blob[m]:
                pred = self.predict(m, n)
                targets.append(r)
                predictions.append(pred)

        return targets, predictions

    def rmse(self, is_test) -> float:
        targets, predictions = self.get_predictions(is_test)
        mse = np.square(np.subtract(targets, predictions)).mean()
        return np.sqrt(mse)

    def get_user_items(self, user_index: int) -> List[float]:
        return [rating[0] for rating in self.user_data_blob[user_index]]

    def get_user_profile(self, user_index: int):
        user_id = self.reverse_user_indexes[user_index]
        ratings = []
        item_indexes = []
        for rating in self.user_data_blob[user_index]:
            ratings.append(rating[1])
            item_indexes.append(rating[0])

        items_details = self.get_items_from_file(item_indexes)
        print(f"UID ::: {user_id}\nHabit:\n")
        print(tuple(zip(items_details[0], items_details[1], ratings)))

    def get_items_from_file(self, item_indexes: List[int]) -> Tuple[List[str], List[str]]:
        item_ids: List[str] = [self.reverse_item_indexes[index] for index in item_indexes]
        item_names = [""] * len(item_indexes)
        with open(self.dataset.items_path, "r", encoding="ISO-8859-1") as file:
            lines = file.readlines()[self.dataset.skip_rows:]
            for ind, line in enumerate(lines):
                content: List[str] = line.split(self.dataset.items_sep)
                if content[0] in item_ids:
                    ind = item_ids.index(content[0])
                    item_names[ind] = content[1]

        return item_ids, item_names

    def get_recommendations_for_user(self, user_index: int, count: int = 10):
        self.get_user_profile(user_index)
        user_item_indexes = [i for i in range(len(self.item_indexes)) if i not in self.get_user_items(user_index)]
        predicted_ratings = []
        for item_index in user_item_indexes:
            predicted_ratings.append(self.predict(user_index, item_index, is_rmse=False))

        top_items = sorted(tuple(zip(user_item_indexes, predicted_ratings)), key=lambda x: x[1], reverse=True)[:count]
        top_item_indexes = [pair[0] for pair in top_items]
        top_item_ratings = [pair[1] for pair in top_items]
        item_ids, item_names = self.get_items_from_file(top_item_indexes)
        return tuple(zip(item_ids, item_names, top_item_ratings))

    def plot_game_feature_vectors_embedded(self, item_per_feature=5, save_figure: bool = True):
        features_items_vectors = get_empty_blob(len(self.feature_indexes))
        markers = [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "s", "p", "P", "*", "h", "+", "x", "d", "D", "H"]
        labels = [key for key in self.feature_name_index]
        plt.figure(figsize=(13, 9))
        for feature_index, feature_items in enumerate(self.feature_items_data_blob):
            if feature_items:
                random_items = sample(feature_items, k=item_per_feature)
                for item_index in random_items:
                    item_vector = self.item_vector[item_index]
                    features_items_vectors[feature_index].append((item_vector[0], item_vector[1], item_index))

        for feature_index, feature_items_vectors in enumerate(features_items_vectors):
            if feature_items_vectors:
                random_text = sample(feature_items_vectors, k=1)
                xs = [feature_items_vector[0] for feature_items_vector in feature_items_vectors]
                ys = [feature_items_vector[1] for feature_items_vector in feature_items_vectors]
                plt.scatter(xs, ys, marker=markers[feature_index], s=50)
                for item_to_text in random_text:
                    id_, name_ = self.get_items_from_file([item_to_text[2]])
                    text = name_[0].split('(')[0]
                    plt.annotate(text, (item_to_text[0], item_to_text[1]))

        plt.grid(color='grey', linestyle='-.', linewidth=0.5, alpha=0.5)
        plt.title(r"Movie feature vectors embedded in $\mathbb{R}^2$, tagged by genre")
        plt.legend(labels, bbox_to_anchor=(1.2, 0.6), loc='center right')
        plt.tight_layout()

        if save_figure:
            plt.savefig(f'{self.fig_dir}item_vectors_embedding.pdf')

        plt.show()
