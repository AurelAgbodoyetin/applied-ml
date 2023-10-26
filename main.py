from data_provider import get_sparse_matrices_and_dump
from datasets import datasets
import multiprocessing as mp
from als_model import ALSModel


if __name__ == "__main__":
    # provider = get_sparse_matrices_and_dump(dataset=datasets['10m'])
    
    test = ALSModel(
        dataset=datasets['10m'],
        # dataset=datasets['100k'],
        biases_only=False,
        use_features=True,
        n_iter=10,
        dims=2,
        tau=.01,
        lambd=.1,
        gamma=.01,
        mu=.0,
        save_figures=True,
        n_jobs=mp.cpu_count(),
    )

    # test.train(parallel=True, plot=True, save_best=True)
    # test.load_parameters()
    # recommendations = test.get_recommendations_for_user(9)
    # print("Recommendations")
    # print(recommendations)
    # test.plot_item_vectors_embedded(save_figure=True)
    # test.plot_feature_vectors_embedded(save_figure=True)
    # test.plot_feature_and_items_vectors_embedded(feature_index=5, save_figure=True)
    # test.get_user_rated_one_item_recommendations(item_index=2)
