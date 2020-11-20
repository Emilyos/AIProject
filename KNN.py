import numpy as np
from scipy.spatial import distance


class KNN:
    def __init__(self, k: int, dist_func=distance.euclidean, stochastic=False):
        self.k = k
        self.train_samples = None
        self.train_labels = None
        self.dist_func = dist_func
        self.classes = None
        self.stochastic = stochastic

    def reset(self):
        self.train_samples = None
        self.train_labels = None

    def fit(self, samples: np.ndarray, labels: np.ndarray):
        assert len(samples) == len(labels)
        assert len(samples) >= self.k
        self.train_samples: np.ndarray = samples
        self.train_labels: np.ndarray = labels.astype(np.int, copy=False)
        self.classes: np.ndarray = np.unique(labels).astype(np.int, copy=False)
        self.classes.sort()

    def _k_nearest(self, sample: np.ndarray):
        dists = np.zeros(shape=self.train_samples.shape[0])
        new_k = 2 * self.k if self.stochastic else self.k
        for i, train_sample in enumerate(self.train_samples):
            dists[i] = self.dist_func(sample, train_sample)
        min_k_indexes = np.argpartition(dists, new_k)
        if self.stochastic:
            return np.random.choice(min_k_indexes, self.k)
        return min_k_indexes[:new_k]

    def _get_votes(self, distances):
        votes = np.zeros_like(self.classes)
        for idx in distances:
            iddd = self.train_labels[idx, 0]
            votes[iddd] += 1
        return votes

    def _majority(self, distances):
        votes = self._get_votes(distances)
        return np.argmax(votes)

    def predict(self, samples: np.ndarray) -> np.ndarray:
        y_predict = np.empty(shape=samples.shape[0])
        for i, sample in enumerate(samples):
            k_nearest = self._k_nearest(sample)
            y_predict[i] = self._majority(k_nearest)
        return y_predict
