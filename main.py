from datasets import datasets
from als_model import ALSModel


if __name__ == "__main__":
    test = ALSModel(
        dataset=datasets['100k_csv'],
        n_iter=10,
        dims=3,
        tau=.1,
        lambd=.01,
        gamma=.01,
        mu=.0,
        save_figures=False,
        n_jobs=8,
    )

    # test.train(parallel=True, plot=True, save_best=True)
    test.load_parameters()
    recommendations = test.get_recommendations_for_user(90)
    print("Recommendations")
    print(recommendations)
