import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from Classifier import Classifer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


class KNN(Classifer):
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

    def train(self, samples: np.ndarray, labels: np.ndarray):
        assert len(samples) == len(labels)
        self.train_samples: np.ndarray = samples
        self.train_labels: np.ndarray = labels.astype(np.int, copy=False)
        self.classes: np.ndarray = np.arange(np.amax(self.train_labels) + 1)

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
            iddd = self.train_labels[idx]
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

    @staticmethod
    def crossValidate(k_fold, X: np.ndarray, y: np.ndarray, params: np.ndarray, stochastic=False,
                      dist_func=distance.euclidean):
        ss = " Stochastic" if stochastic else ""
        print(f"{k_fold}-Fold Cross Validating{ss} KNN Model with K's={params}")
        params = np.array(params)
        accuracies = np.zeros(shape=params.shape)
        for j, k in enumerate(params):
            sub_accuracies = np.zeros(shape=k_fold)
            splitter = StratifiedKFold(n_splits=k_fold)
            print(f"Running with K={k}")
            classifier = KNN(k=k, stochastic=stochastic, dist_func=dist_func)
            i = 0
            World = f"{i}'th Fold"
            for train_index, test_index in tqdm(splitter.split(X, y), total=splitter.get_n_splits()):
                x_train, y_train = X[train_index], y[train_index]
                x_test, y_test = X[test_index], y[test_index]
                classifier.train(x_train, y_train)
                predicts = classifier.predict(x_test)
                accuracy = accuracy_score(y_test, predicts) * 100
                sub_accuracies[i] = accuracy
                i += 1

            avg_accuracy = np.average(sub_accuracies)
            print(f"AVG Accuracy: {avg_accuracy:2.3f}%")
            accuracies[j] = avg_accuracy
        best_k = params[accuracies.argmax()]
        print(f"Best performance: {np.amax(accuracies):2.3f}% for k={best_k}")
        return best_k, [(params[i], accuracies[i]) for i in range(len(params))]

    def getClassifierName(self):
        return f"KNNClassifier"
