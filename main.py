from typing import List
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

if __name__ == "__main__":
    # parallel.compute_using_threading(epochs, latent_dims, tau_, lambda_, gamma_)
    # parallel.compute_using_multiprocessing(epochs, latent_dims, tau_, lambda_, gamma_)

    test_parallel = SparseMatrix(
        data_path=data_20m_path,
        sep=sep_20m,
        skip_rows=skip_rows_20m,
        split_ratio=.2,
        n_iter=1,
        dims=3,
        tau=.01,
        lambd=.01,
        gamma=.01,
        mu=.0,
        save_figures=True,
        parallel_extraction=False,
        parallel_indexing=False,
        show_power_law=False
    )

    test_parallel = SparseMatrix(
        data_path=data_20m_path,
        sep=sep_20m,
        skip_rows=skip_rows_20m,
        split_ratio=.2,
        n_iter=1,
        dims=3,
        tau=.01,
        lambd=.01,
        gamma=.01,
        mu=.0,
        save_figures=True,
        parallel_extraction=True,
        parallel_indexing=False,
        show_power_law=False
    )

    print("Done")
    pass


    # test.perform_als(is_parallel=False)
    # test.perform_als(is_parallel=False)
