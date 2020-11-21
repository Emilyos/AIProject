import numpy as np
import pandas as pd
import KNN
import matplotlib.pyplot as plt
from Dataloader import Dataloader
from ForestClassifier import ForestClassifier
from ID3 import ID3, FeatureType, Feature


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
                        categorical_features=[1], dataset_name="fifa19")
    # Select dataset
    current_ds = bena_ds
    # Extract train/test
    X_train, y_train = current_ds.train_samples(normlize=True, frac=1, shuffle=False)
    X_test, y_test = current_ds.test_samples(normlize=True)

    # find best k parameter for the model
    params = np.array([1, 5, 10, 20])
    features = [Feature(i, FeatureType.Continuous) for i in range(8)]
    best_k = ID3.crossValidate(2, X_train, y_train, params=params, features=features)
    return best_k


if __name__ == "__main__":
    bena_ds = Dataloader("datasets/236501/train.csv", "datasets/236501/test.csv")
    fifa19 = Dataloader("datasets/fifa19/train.csv", "datasets/fifa19/test.csv", index_col=0,
                        categorical_features=[1], dataset_name="fifa19")
    performances = []
    best_m = 100
    features = [Feature(i, FeatureType.Continuous) for i in range(fifa19.n_features - 1)]
    features[1] = Feature(1, FeatureType.Discrete, np.arange(26))
    for i in range(1, 22, 2):
        forest = ForestClassifier(i, fifa19, ID3, {"min_leaf_samples": best_m, "features": features}, normlize=False)
        forest.buildForest(frac=2 / 3)
        performances.append((forest.performance()))

    plt.plot([ele[0] for ele in performances], [ele[1] for ele in performances])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    plt.show()
