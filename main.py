import numpy as np
import pandas as pd
import ID3
import KNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance


def fifa_players_dist(sample1, sample2):
    diff = np.abs(sample1 - sample2)
    if sample1[1] == sample2[1]:
        diff[1] = 0
    else:
        diff[1] = 1
    return np.sum(diff ** 2) ** 0.5


def main():
    train, test = pd.read_csv("datasets/fifa19/train.csv", index_col=0).to_numpy(), pd.read_csv(
        "datasets/fifa19/test.csv", index_col=0).to_numpy()
    scaller = MinMaxScaler()

    train_x, train_y = train[:, :-1], train[:, -1]
    test_x: np.ndarray = test[:, :-1]
    test_y: np.ndarray = test[:, -1]

    train_pos_col = train_x[:, 1]
    test_pos_col = test_x[:, 1]
    train_x = scaller.fit_transform(train_x)
    test_x = scaller.fit_transform(test_x)
    train_x[:, 1] = train_pos_col
    test_x[:, 1] = test_pos_col

    k_to_check = np.array([133, 13])
    best_k = KNN.KNN.crossValidate(5, train_x, train_y, k_to_check)
    print(f"Runing Cross-Validation with normalization!")
    knnClassifier = KNN.KNN(k=best_k)
    knnClassifier.train(train_x, train_y)
    predicits = knnClassifier.predict(test_x)
    print(f"Model Performance {accuracy_score(test_y, predicits) * 100}%")


if __name__ == "__main__":
    main()
