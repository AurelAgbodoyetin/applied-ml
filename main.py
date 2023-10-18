from typing import List
from sparse_matrix import SparseMatrix

latent_dims: List[int] = [3, 5, 10, 20, 50, 100]
tau_: float = .1
epochs: int = 30
lambda_: float = .1
gamma_: float = .1

if __name__ == "__main__":
    test = SparseMatrix(
        dataset_name="100k",
        n_iter=15,
        dims=4,
        tau=.01,
        lambd=.01,
        gamma=.01,
        mu=.0,
        save_figures=False,
        n_jobs=8,
    )

    test.perform_als(parallel=True, dims=2)
