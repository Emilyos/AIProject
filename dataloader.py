import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import ID3
import KNN


def id3():
    x_train = pd.read_csv("datasets/fifa19/train.csv", index_col=0)
    test = pd.read_csv("datasets/fifa19/test.csv", index_col=0)
    print(x_train)
    print(test)
    features = np.array([ID3.Feature(index=i, type=ID3.FeatureType.Continuous) for i in range(x_train.shape[1] - 1)])
    features[1] = ID3.Feature(index=1, type=ID3.FeatureType.Discrete, domain=list(range(26)))

    dt = ID3.ID3(100, stochastic=True)
    x_train = x_train.to_numpy()
    test = test.to_numpy()
    y_train: np.ndarray = x_train[:, -1]
    y_train = y_train.reshape(-1, 1)
    y_train = y_train.astype(int, copy=False)
    x_train = np.delete(x_train, -1, axis=1)

    y_test: np.ndarray = test[:, -1]
    y_test = y_test.reshape(-1, 1)
    y_test = y_test.astype(int, copy=False)
    x_test = np.delete(test, - 1, axis=1)

    # print(csv)
    dt.fit(x_train, y_train, features)

    predicts = dt.predict(x_test)
    y_test = y_test.flatten()
    correct = 0
    print(len(predicts), "   ", len(y_test))
    for i in range(len(y_test)):
        if (predicts[i] == y_test[i]):
            correct += 1
    accuracy = (correct / len(y_test)) * 100
    print(accuracy)


def knn():
    knnClassifier = KNN.KNN(k=3, stochastic=False)
    train, test = pd.read_csv("datasets/236501/train.csv").to_numpy(), pd.read_csv(
        "datasets/236501/test.csv").to_numpy()
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x: np.ndarray = test[:, :-1]
    test_y: np.ndarray = test[:, -1]
    print(train_y)
    knnClassifier.train(train_x, train_y)
    for k in range(1):
        y_predict = knnClassifier.predict(test_x)
        test_y = test_y.flatten()
        correct = 0
        for i in range(len(test_y)):
            if test_y[i] == y_predict[i]:
                correct += 1
        print(f"itr={k} @ Accuracy={(correct / len(test_y)) * 100}")


if __name__ == "__main__":
    # knn()
    pass
