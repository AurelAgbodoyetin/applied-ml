from typing import List
import parallel
from sparse_matrix import SparseMatrix

big_data_path = "data/25m/ratings.csv"
small_data_path = "data/100k/u.data"
joblib_dumps_path = "joblib_dumps"

latent_dims: List[int] = [3, 5, 10, 20, 50, 100]
tau_: float = .1
epochs: int = 30
lambda_: float = .1
gamma_: float = .1

if __name__ == "__main__":
    # parallel.compute_using_threading(epochs, latent_dims, tau_, lambda_, gamma_)
    # parallel.compute_using_multiprocessing(epochs, latent_dims, tau_, lambda_, gamma_)
    
    # test = SparseMatrix(
    #     data_path=small_data_path,
    #     sep="\t",
    #     split_ratio=.2,
    #     n_iter=10,
    #     dims=5,
    #     tau=.01,
    #     lambd=.01,
    #     gamma=.01,
    #     mu=.0,
    #     save_figures=True,
    # )
    # test.perform_als(dims=3, tau=0.01, lambd=0.01, gamma=0.01)


