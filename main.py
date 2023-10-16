from typing import List

from data_provider import DataProvider
from sparse_matrix import SparseMatrix

data_25m_path: str = "../datasets/25M/ratings.csv"
sep_25m: str = ","
skip_rows_25m: int = 1
data_20m_path: str = "../datasets/20M/ratings.csv"
sep_20m: str = ","
skip_rows_20m: int = 1
data_10m_path: str = "../datasets/10M/ratings.dat"
sep_10m: str = "::"
skip_rows_10m: int = 0
data_100k_path: str = "data/100k/u.data"
sep_100k: str = "\t"
skip_rows_100k: int = 0

joblib_dumps_path = "joblib_dumps"

latent_dims: List[int] = [3, 5, 10, 20, 50, 100]
tau_: float = .1
epochs: int = 30
lambda_: float = .1
gamma_: float = .1


def get_sparse_matrices_and_dump():
    data_provider: DataProvider = DataProvider(
        data_path=data_20m_path,
        sep=sep_20m,
        skip_rows=skip_rows_20m,
        split_ratio=.2,
        dump=True,
        show_power_law=False,
        save_figures=False,
        parallel=True,
    )
    data_provider.extract_all_file_ds()
    data_provider.extract_test_set_ds()
    data_provider.extract_train_set_ds()

    data_provider.plot_all_file_ds()
    return data_provider


if __name__ == "__main__":
    # parallel.compute_using_threading(epochs, latent_dims, tau_, lambda_, gamma_)
    # parallel.compute_using_multiprocessing(epochs, latent_dims, tau_, lambda_, gamma_)

    get_sparse_matrices_and_dump()

    test = SparseMatrix(
        n_iter=1,
        dims=3,
        tau=.01,
        lambd=.01,
        gamma=.01,
        mu=.0,
        save_figures=False,
    )
    print("Done")
    pass

    # test.perform_als(is_parallel=False)
    # test.perform_als(is_parallel=False)
