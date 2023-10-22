from datasets import datasets
import multiprocessing as mp
from als_model import ALSModel


if __name__ == "__main__":
    test = ALSModel(
        dataset=datasets['1m'],
        # dataset=datasets['100k_csv'],
        biases_only=False,
        use_features=True,
        n_iter=30,
        dims=3,
        tau=.1,
        lambd=.01,
        gamma=.01,
        mu=.0,
        save_figures=True,
        n_jobs=mp.cpu_count(),
    )

    # test.train(parallel=True, plot=True, save_best=True)
    test.load_parameters()
    # recommendations = test.get_recommendations_for_user(90)
    # print("Recommendations")
    # print(recommendations)
    test.plot_game_feature_vectors_embedded(save_figure=True)
