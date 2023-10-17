import threading
import multiprocessing as mp
from typing import List
from timebudget import timebudget

from sparse_matrix import SparseMatrix


def get_sparse_matrices(data_path: str, epochs: int, dims: List[int], tau_: float, lambda_: float, gamma_: float):
    return [
        SparseMatrix(
            data_path=data_path,
            sep="\t",
            split_ratio=.2,
            n_iter=epochs,
            dims=dim,
            tau=tau_,
            lambd=lambda_,
            gamma=gamma_,
            mu=.0,
            save_figures=False,
            show_power_law=False
        ) for dim in dims
    ]


@timebudget
def compute_using_threading(data_path: str, epochs: int, dims: List[int], tau_: float, lambda_: float, gamma_: float):
    sparse_matrices: List[SparseMatrix] = get_sparse_matrices(data_path, epochs, dims, tau_, lambda_, gamma_)

    threads: List[threading.Thread] = [
        threading.Thread(target=matrix.perform_als) for matrix in sparse_matrices
    ]

    for thread in threads:
        thread.start()


@timebudget
def compute_using_multiprocessing(data_path: str, epochs: int, dims: List[int], tau_: float, lambda_: float,
                                  gamma_: float):
    sparse_matrices: List[SparseMatrix] = get_sparse_matrices(data_path, epochs, dims, tau_, lambda_, gamma_)

    processes: List[mp.Process] = [
        mp.Process(target=matrix.perform_als) for matrix in sparse_matrices
    ]

    for process in processes:
        process.start()


if __name__ == "__main__":
    pass
