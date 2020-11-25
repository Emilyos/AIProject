import numpy as np
import pandas as pd
from Utils import *
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean


class Dataloader:

    def __init__(self, train_filename, test_filename, index_col=None, class_idx=-1, categorical_features=[],
                 dist_func=euclidean):
        self.dist_func = dist_func
        self.train_csv = pd.read_csv(train_filename, index_col=index_col)
        self.test_csv = pd.read_csv(test_filename, index_col=index_col)
        self.class_idx = class_idx
        self.categorical_features = categorical_features
        self.n_features = self.train_csv.shape[1] - 1
        self.n_test_samples = self.test_csv.to_numpy().shape[0]
        self.n_train_samples = self.train_csv.to_numpy().shape[0]

    def _parse(self, pd_dataset, normlize, frac=1, shuffle=False):
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

    def _findDomain(self, feature_idx):
        train_d = set(self.train_csv.iloc[:, feature_idx])
        test_d = set(self.test_csv.iloc[:, feature_idx])
        return np.array(list(train_d.union(test_d)))

    def getFeatures(self):
        features = []
        for i in range(self.n_features):
            if i in self.categorical_features:
                features.append(Feature(index=i, type=FeatureType.Discrete, domain=self._findDomain(i)))
            else:
                features.append(Feature(index=i, type=FeatureType.Continuous))
        return features
