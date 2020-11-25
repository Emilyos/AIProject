import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class ForestClassifier:

    def __init__(self, n_trees, dataloader, classifier_ctr, classifier_params, normlize=False):
        self.classifier_params = classifier_params
        self.dataloader = dataloader
        self.n_trees = n_trees
        self.classifier = classifier_ctr
        self.trees = []
        self.normlize = normlize

    def buildForest(self, frac):
        self.trees = []
        for i in range(self.n_trees):
            self.trees.append(self.classifier(**self.classifier_params))
            train_X, train_y = self.dataloader.train_samples(normlize=self.normlize, frac=frac, shuffle=True)
            self.trees[i].train(train_X, train_y)

    def performance(self):
        test_X, test_y = self.dataloader.test_samples(normlize=self.normlize)
        n_samples = test_X.shape[0]
        results = np.empty(shape=(self.n_trees, n_samples))
        for i, tree in tqdm(enumerate(self.trees), total=len(self.trees)):
            results[i] = tree.predict(test_X)
        votes = np.empty(shape=test_X.shape[0])
        for i in range(n_samples):
            votes[i] = np.bincount(results[:, i].astype(np.int)).argmax()
        accuracy = accuracy_score(test_y, votes) * 100
        return self.n_trees, accuracy, votes
