import enum
from tqdm import tqdm
import numpy as np


class FeatureType(enum.Enum):
    Discrete = 0
    Continuous = 1


class Feature:
    def __init__(self, index, type, domain=[], used=False):
        self.index = index
        self.type = type
        self.domain = domain
        self.used = False


class ID3:
    class Node:

        def __init__(self, clf, is_leaf=False, feature: Feature = None, split_val=None):
            self.is_leaf = is_leaf
            self.pointers = dict()
            self.feature = feature
            self.clf = clf
            self.split_val = split_val

        def addChild(self, val, node):
            self.pointers[val] = node

        def setAsLeaf(self):
            self.is_leaf = True
            self.pointers.clear()

        def isLeaf(self):
            return self.is_leaf

        def getClassification(self):
            return self.clf

        def travel(self, sample: np.ndarray):
            val = sample[self.feature.index]
            if self.feature.type == FeatureType.Discrete:
                return self.pointers.get(val)

            assert self.split_val is not None
            if val >= self.split_val:
                return self.pointers['ge']
            else:
                return self.pointers['less']

    def __init__(self, min_leaf_samples: int, stochastic: bool = False):
        self.features_types = np.array([])
        self.min_leaf_samples = min_leaf_samples
        self.stochastic = stochastic
        self.root = None

    def Entropy(self, x_train: np.ndarray, y_train: np.ndarray):
        if not len(x_train):
            print("#samples = 0!")
            return +np.inf  # no
        classes = np.unique(y_train.flatten())
        sum = 0
        n_samples = len(x_train)
        flattened_y = y_train.flatten()
        for i in classes:
            p = ((flattened_y == i).sum()) / n_samples
            if p > 0:
                sum += (p * np.log2(p))
        return -sum

    def _get_IG(self, feature: Feature, x_train: np.ndarray, y_train: np.ndarray):
        E = len(x_train)
        entropy = self.Entropy(x_train, y_train)

        if feature.type == FeatureType.Discrete:  # Discrete values
            values = np.unique(x_train[:, feature.index])
            information_gain = entropy
            for value in values:
                mask = x_train[:, feature.index] == value
                x_i = x_train[mask, :]
                y_i = y_train[mask, :]
                E_i_entropy = self.Entropy(x_i, y_i)
                information_gain -= (len(x_i) / E) * E_i_entropy
            return information_gain, None
        else:  # Continuous values
            v = np.sort(np.unique(x_train[:, feature.index]))
            values = [(v[i] + v[i + 1]) / 2 for i in range(len(v) - 1)]
            max_information_gain = -np.inf
            split_value = None
            for value in values:
                less_mask = x_train[:, feature.index] < value
                less_x_train = x_train[less_mask, :]
                less_y_train = y_train[less_mask, :]
                ge_mask = x_train[:, feature.index] >= value  # greater equal than value samples.
                ge_x_train = x_train[ge_mask, :]
                ge_y_train = y_train[ge_mask, :]

                child_entropy = (len(less_x_train) / E) * self.Entropy(less_x_train, less_y_train)
                child_entropy += (len(ge_x_train) / E) * self.Entropy(ge_x_train, ge_y_train)
                information_gain = entropy - child_entropy
                split_value = value if information_gain > max_information_gain else split_value
                max_information_gain = information_gain if information_gain > max_information_gain else max_information_gain

            return max_information_gain, split_value

    def _getNextFeature(self, x_train: np.ndarray, y_train: np.ndarray, features: np.ndarray):
        info_gains = np.zeros(shape=features.shape)
        split_val = np.zeros(shape=features.shape)
        for i, feature in enumerate(features):
            if feature.type == FeatureType.Discrete and feature.used: continue
            info_gains[i], split_val[i] = self._get_IG(feature, x_train, y_train)

        info_gains[info_gains < 0] = 0
        info_sum = sum(info_gains)
        if info_sum == 0.0:
            print("info_sum is zero")
        if self.stochastic:
            if info_sum == 0:
                print("it's 0, yeah... stochastics")
            P = np.array([info_gains[i] / info_sum for i in range(len(info_gains))])
            idx = np.random.choice(np.arange(0, len(info_gains)), p=P)
        else:
            idx = info_gains.argmax()

        return features[idx], split_val[idx]

    def build_recursively(self, x_train: np.ndarray, y_train: np.ndarray, features, default_clf):
        if len(x_train) == 0:
            return self.Node(is_leaf=True, clf=default_clf)
        clf = np.bincount(y_train.flatten()).argmax()
        if len(x_train) < self.min_leaf_samples:
            return self.Node(is_leaf=True, clf=clf)
        if len(features) == 0 or len(np.unique(y_train.flatten())) == 1:
            return self.Node(is_leaf=True, clf=clf)

        f, split_val = self._getNextFeature(x_train, y_train, features)
        if f.type == FeatureType.Discrete:
            f.used = True
        node = self.Node(clf, feature=f)
        if f.type == FeatureType.Discrete:
            for d in f.domain:
                child_samples_mask = x_train[:, f.index] == d
                child_x = x_train[child_samples_mask, :]
                child_y = y_train[child_samples_mask, :]
                child_node = self.build_recursively(x_train=child_x, y_train=child_y, features=features,
                                                    default_clf=clf)
                node.addChild(d, child_node)
        else:
            left_child_mask = x_train[:, f.index] < split_val
            right_child_mask = x_train[:, f.index] >= split_val
            left_child_x = x_train[left_child_mask, :]
            left_child_y = y_train[left_child_mask, :]
            right_child_x = x_train[right_child_mask, :]
            right_child_y = y_train[right_child_mask, :]
            left_node = self.build_recursively(x_train=left_child_x, y_train=left_child_y, features=features,
                                               default_clf=clf)
            right_node = self.build_recursively(x_train=right_child_x, y_train=right_child_y, features=features,
                                                default_clf=clf)
            node.addChild('less', left_node)
            node.addChild('ge', right_node)
            node.split_val = split_val
        return node

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, features):
        clf = np.bincount(y_train.flatten()).argmax()
        self.root = self.build_recursively(x_train, y_train, features, clf)

    def predict_(self, sample):
        current_node: ID3.Node = self.root
        while not current_node.isLeaf():
            next_node = current_node.travel(sample)
            if next_node == None:
                return current_node.getClassification()
            current_node = next_node
        return current_node.getClassification()

    def predict(self, samples: np.ndarray):
        prediction = np.empty(shape=samples.shape[0])
        for i, sample in tqdm(enumerate(samples)):
            prediction[i] = self.predict_(sample)

        return prediction
