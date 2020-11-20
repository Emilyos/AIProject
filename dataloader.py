import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import ID3

if __name__ == "__main__":
    csv = pd.read_csv("datasets/fifa19/players.csv",index_col=0)
    x_train, test = train_test_split(csv, test_size=(1 / 3), shuffle=False)
    # x_train = csv
    # test = pd.read_csv("datasets/236501/test.csv")
    features = np.array([ID3.Feature(index=i, type=ID3.FeatureType.Continuous) for i in range(csv.shape[1] - 1)])
    features[1] = ID3.Feature(index=1, type=ID3.FeatureType.Discrete, domain=list(range(26)))

    dt = ID3.ID3(100,stochastic=True)
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

    print(csv)
    dt.fit(x_train, y_train, features)



    predicts = dt.predict(x_test)
    y_test = y_test.flatten()
    correct = 0
    print(len(predicts),"   ",len(y_test))
    for i in range(len(y_test)):
        if (predicts[i] == y_test[i]):
            correct += 1
    accuracy = (correct/len(y_test))*100
    print(accuracy)