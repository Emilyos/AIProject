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


def main():
    bena_ds = Dataloader("datasets/236501/train.csv", "datasets/236501/test.csv", dataset_name="bena")
    fifa19 = Dataloader("datasets/fifa19/train.csv", "datasets/fifa19/test.csv", index_col=0,
                        categorical_features=[1], dataset_name="fifa19", dist_func=fifa_players_dist)
    # Select dataset
    current_ds = bena_ds
    # Extract train/test
    X_train, y_train = current_ds.train_samples(normlize=True, frac=1, shuffle=False)
    X_test, y_test = current_ds.test_samples(normlize=True)

    # find best k parameter for the model
    params = np.array([1, 5, 10, 20])
    # best_k = ID3.crossValidate(2, X_train, y_train, params=params, features=features)
    return best_k


if __name__ == "__main__":
    bena_ds = Dataloader("datasets/236501/train.csv", "datasets/236501/test.csv", dataset_name="263501")
    fifa19 = Dataloader("datasets/fifa19/train.csv", "datasets/fifa19/test.csv", index_col=0,
                        categorical_features=[1], dataset_name="fifa19")

    best_k = Utils.CrossValidateKNN(3, bena_ds, [1, 3, 7, 9])
    Utils.run_experiment(2, bena_ds, KNN, {"k": best_k})
