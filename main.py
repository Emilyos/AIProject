import numpy as np
from KNN import KNN
import Utils

from Dataloader import Dataloader


def fifa_players_dist(sample1, sample2):
    diff = np.abs(sample1 - sample2)
    if sample1[1] == sample2[1]:
        diff[1] = 0
    else:
        diff[1] = 1
    return np.sum(diff ** 2) ** 0.5


if __name__ == "__main__":
    bena_ds = Dataloader("datasets/236501/train.csv", "datasets/236501/test.csv", dataset_name="263501")
    fifa19 = Dataloader("datasets/fifa19/train.csv", "datasets/fifa19/test.csv", index_col=0,
                        categorical_features=[1], dataset_name="fifa19", dist_func=fifa_players_dist)

    # best_k = Utils.CrossValidateKNN(3, bena_ds, [1, 3, 7, 9])
    # Utils.run_experiment(2, bena_ds, KNN, {"k": best_k})

    best_k = Utils.CrossValidateKNN(10, fifa19, [3, 9, 13, 19, 33, 133], stochastic=False, normlize=True)
    Utils.run_experiment(3, fifa19, KNN, {"k": best_k, "dist_func": fifa19.dist_func}, normlize=True)

    # best_k = Utils.CrossValidateKNN(10, fifa19, [3, 9, 13, 19, 33, 133], stochastic=True, normlize=True)
    # Utils.run_experiment(4, fifa19, KNN, {"k": best_k, "dist_func": fifa19.dist_func, "stochastic": True},
    #                      normlize=True)
