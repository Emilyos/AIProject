import enum

import numpy as np


class FeatureType(enum.Enum):
    Discrete = 0
    Continuous = 1


class Feature:
    index: int
    type: FeatureType
    domain: np.ndarray


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
                return val < self.pointers['less']

    def __init__(self, min_leaf_samples: int, stochastic: bool = False):
        self.features_types = np.array([])
        self.min_leaf_samples = min_leaf_samples
        self.stochastic = stochastic
        self.root = None

    def Entropy(self, samples: np.ndarray):
        if not len(samples):
            print("#samples = 0!")
            return None
        classes = np.unique(samples[:, -1])
        sum = 0
        for i in classes:
            p = ((samples[:, -1] == i).sum()) / len(samples)
            if p > 0:
                sum += (p * np.log2(p))
        return -sum

    def _get_IG(self, feature: Feature, samples_in_node: np.ndarray):
        E = len(samples_in_node)
        entropy = self.Entropy(samples_in_node)

        if feature.type == FeatureType.Discrete:  # Discrete values
            values = samples_in_node[:, feature.index]
            information_gain = entropy
            for value in values:
                mask = samples_in_node[:, feature.index] == value
                E_i = samples_in_node[mask, :]
                E_i_entropy = self.Entropy(E_i)
                information_gain -= (len(E_i) / E) * E_i_entropy
            return information_gain, None
        else:  # Continuous values
            values = samples_in_node[:, feature.index]
            max_information_gain = -np.inf
            split_value = None
            for value in values:
                less_mask = samples_in_node[:, feature.index] < value
                less_samples = samples_in_node[less_mask, :]
                ge_mask = samples_in_node[:, feature.index] >= value  # greater equal than value samples.
                ge_samples = samples_in_node[ge_mask, :]
                child_entropy = (len(less_samples) / E) * self.Entropy(less_samples)
                child_entropy += (len(ge_samples) / E) * self.Entropy(ge_samples)
                information_gain = entropy - child_entropy
                max_information_gain = information_gain if information_gain > max_information_gain else max_information_gain
                split_value = value if information_gain > max_information_gain else split_value
            return max_information_gain, split_value

    def build_recursively(self, E: np.ndarray, features: list[Feature], default_clf):
        if len(E) == 0:
            return self.Node(is_leaf=True, clf=default_clf)
        clf = np.bincount(E[:, -1]).argmax()
        if len(E) < self.min_leaf_samples:
            return self.Node(is_leaf=True, clf=clf)
        if len(features) == 0 or len(np.unique(E[:, -1])) == 1:
            return self.Node(is_leaf=True, clf=clf)
        f, f_ig, split_val = None, -np.inf, None

        for feature in features:
            ig, val = self._get_IG(feature, E)
            f = feature if (ig > f_ig) else f
            split_val = val if (ig > f_ig) else split_val
            f_ig = ig if (ig > f_ig) else f_ig

        if f.type == FeatureType.Discrete:
            np.delete(features, f)
        node = self.Node(clf, feature=f)
        if f.type == FeatureType.Discrete:
            for d in f.domain:
                child_samples_mask = E[:, f.index] == d
                child_samples = E[child_samples_mask, :]
                child_node = self.build_recursively(E=child_samples, features=features, default_clf=clf)
                node.addChild(d, child_node)
        else:
            left_child_mask = E[:, f.index] < split_val
            right_child_mask = E[:, f.index] >= split_val
            left_child_samples = E[left_child_mask, :]
            right_child_samples = E[right_child_mask, :]
            left_node = self.build_recursively(E=left_child_samples, features=features, default_clf=clf)
            right_node = self.build_recursively(E=right_child_samples, features=features, default_clf=clf)
            node.addChild('less', left_node)
            node.addChild('ge', right_node)
            node.split_val = split_val
        return node

    def fit(self, samples: np.ndarray, features: list[Feature]):
        clf = np.bincount(E[:, -1]).argmax()
        self.root = self.build_recursively(samples, features, clf)

    def predict_(self, sample):
        current_node: ID3.Node = root
        while not current_node.isLeaf():
            current_node = current_node.travel(sample)
        return current_node.getClassification()

    def predict(self, samples: np.ndarray):
        prediction = np.empty(shape=samples.shape[0])
        for i, sample in enumerate(samples):
            prediction[i] = self.predict_(sample)
        return prediction
