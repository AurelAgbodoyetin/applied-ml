from data_provider import get_sparse_matrices_and_dump
from datasets import datasets
import multiprocessing as mp
from als_model import ALSModel

def search_best_params():
    ks = [i for i in range(1, 2)]
    gammas = [0.1 * i for i in range(1, 5)]
    taus = [0.01 * i for i in range(1, 5)]
    lambdas = [0.02 * i for i in range(1, 5)]
    for k in ks:
        for g in gammas:
            for t in taus:
                for l in lambdas:
                    test = ALSModel(
                        dataset=datasets['1m'],
                        biases_only=False,
                        use_features=False,
                        n_iter=10,
                        dims=k,
                        tau=t,
                        lambd=l,
                        gamma=g,
                        mu=.0,
                        monitor=True,
                        save_figures=False,
                        n_jobs=mp.cpu_count(),
                    )
                    test.train(parallel=True, plot=False, save_best=False)


if __name__ == "__main__":
    provider = get_sparse_matrices_and_dump(dataset=datasets['1m'])

    test = ALSModel(
        dataset=datasets['10m'],
        biases_only=False,
        use_features=True,
        n_iter=10,
        dims=3,
        tau=.03,
        lambd=.02,
        gamma=.4,
        mu=.0,
        monitor=False,
        save_figures=True,
        save_history=True,
        n_jobs=mp.cpu_count(),
    )

    test.train(parallel=True, plot=True, save_best=True)
    # test.load_parameters()

    recommendations = test.get_recommendations_for_user(9)
    print("Recommendations")
    print(recommendations)
    test.get_user_rated_one_item_recommendations(item_index=2)

    test.plot_item_vectors_embedded(save_figure=True)
    test.plot_feature_items_vectors_embedded(save_figure=True, feature_index=2)

    test.plot_feature_vectors_embedded(save_figure=True)
    test.plot_feature_and_items_vectors_embedded(feature_index=5, save_figure=True)
