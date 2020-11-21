import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Dataloader:

    def __init__(self, train_filename, test_filename, index_col=None, class_idx=-1, categorical_features=[],
                 dataset_name=""):
        self.dataset_name = dataset_name
        self.train_csv = pd.read_csv(train_filename, index_col=index_col)
        self.test_csv = pd.read_csv(test_filename, index_col=index_col)
        self.class_idx = class_idx
        self.categorical_features = categorical_features
        self.n_features = self.train_csv.shape[1]

    def _parse(self, pd_dataset, normlize, frac=1, shuffle=False):
        train_np = pd_dataset
        if shuffle:
            train_np = pd_dataset.sample(frac=frac).to_numpy()
        else:
            n_samples = int(pd_dataset.shape[0] * frac)
            train_np = pd_dataset.to_numpy()[:n_samples + 1, :]
        X = train_np[:, :self.class_idx]
        y = train_np[:, self.class_idx]
        if self.class_idx != -1 or self.class_idx == train_np.shape[1]:
            X = np.concatenate((X, train_np[:, self.class_idx + 1:]))
        if normlize:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            for i in self.categorical_features:
                X[:, i] = train_np[:, i]
        return X, y.astype(np.int)

    def train_samples(self, normlize=False, frac=1, shuffle=False):
        return self._parse(self.train_csv, normlize=normlize, frac=frac, shuffle=shuffle)

    def test_samples(self, normlize=False, frac=1, shuffle=False):
        return self._parse(self.test_csv, normlize=normlize, frac=frac, shuffle=shuffle)

    def datasetName(self):
        return self.dataset_name
# def id3():
#     x_train = pd.read_csv("datasets/fifa19/train.csv", index_col=0)
#     test = pd.read_csv("datasets/fifa19/test.csv", index_col=0)
#     print(x_train)
#     print(test)
#     features = np.array([ID3.Feature(index=i, type=ID3.FeatureType.Continuous) for i in range(x_train.shape[1] - 1)])
#     features[1] = ID3.Feature(index=1, type=ID3.FeatureType.Discrete, domain=list(range(26)))
#
#     dt = ID3.ID3(100, stochastic=True)
#     x_train = x_train.to_numpy()
#     test = test.to_numpy()
#     y_train: np.ndarray = x_train[:, -1]
#     y_train = y_train.reshape(-1, 1)
#     y_train = y_train.astype(int, copy=False)
#     x_train = np.delete(x_train, -1, axis=1)
#
#     y_test: np.ndarray = test[:, -1]
#     y_test = y_test.reshape(-1, 1)
#     y_test = y_test.astype(int, copy=False)
#     x_test = np.delete(test, - 1, axis=1)
#
#     # print(csv)
#     dt.fit(x_train, y_train, features)
#
#     predicts = dt.predict(x_test)
#     y_test = y_test.flatten()
#     correct = 0
#     print(len(predicts), "   ", len(y_test))
#     for i in range(len(y_test)):
#         if (predicts[i] == y_test[i]):
#             correct += 1
#     accuracy = (correct / len(y_test)) * 100
#     print(accuracy)
#
