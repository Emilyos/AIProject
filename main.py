import numpy as np
import optparse
import argparse
from KNN import KNN
from ID3 import ID3
import Utils
import os
import sys
import pandas as pd
from Dataloader import Dataloader
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from abc import *


class Dataset(ABC):

    @abstractmethod
    def __init__(self, dataset_name, knn_k, min_leaf_samples):
        path_to_dir = f"datasets/{dataset_name}"
        self.test_path = f"{path_to_dir}/test.csv"
        self.train_path = f"{path_to_dir}/train.csv"
        if not os.path.isdir(path_to_dir):
            print(f"Could not load dataset, directory {path_to_dir} does not exist")
            sys.exit()
        if not os.path.isfile(self.test_path) or not os.path.isfile(self.train_path):
            print(f"File {self.train_path} or {self.test_path} does not exist")
            sys.exit()
        self.knn_k = knn_k
        self.min_leaf_samples = min_leaf_samples
        self.dataset_name = ""
        self.dataloader = None

    def getDatasetName(self):
        return self.dataset_name

    def getDL(self):
        return self.dataloader

    def getKNNParams(self, stochastic, crossValidate=False, k_to_check=[3, 9, 13, 17]):
        if crossValidate:
            best_k = Utils.CrossValidateKNN(crossValidate, self.dataloader, k_to_check=k_to_check,
                                            stochastic=stochastic,
                                            normlize=True, dataset_name=self.dataset_name)
        else:
            best_k = self.knn_k
        return {"k": best_k, "stochastic": stochastic, "dist_func": self.dataloader.dist_func}

    def getID3Params(self, stochastic, crossValidate, m_to_check=[10, 30, 50, 100]):
        if crossValidate:
            best_m = Utils.CrossValidateID3(crossValidate, self.dataloader, m_to_check=m_to_check,
                                            stochastic=stochastic, normlize=False, dataset_name=self.dataset_name)
        else:
            best_m = self.min_leaf_samples  # default
        return {"min_leaf_samples": best_m, "features": self.dataloader.getFeatures(), "stochastic": stochastic}


class Fifa19(Dataset):

    @staticmethod
    def fifa_players_dist(sample1, sample2):
        diff = np.abs(sample1 - sample2)
        if sample1[1] == sample2[1]:
            diff[1] = 0
        else:
            diff[1] = 1
        return np.sum(diff ** 2) ** 0.5

    def __init__(self, dataset_name, knn_k=9, min_leaf_samples=50):
        super().__init__(dataset_name, knn_k, min_leaf_samples)
        self.dataset_name = "fifa19"
        self.dataloader = Dataloader(self.train_path, self.test_path,
                                     index_col=0, dist_func=Fifa19.fifa_players_dist, categorical_features=[1])


class Cancer(Dataset):

    def __init__(self, dataset_name, knn_k, min_leaf_samples, desc=None):
        super().__init__(dataset_name, knn_k, min_leaf_samples)
        self.dataset_name = "Breast Cancer"
        self.dataloader = Dataloader(self.train_path, self.test_path, index_col=0)


def main():
    # parser = optparse.OptionParser()
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp", type=int, help="Experiment number [1-5]", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset to use: [fifa19, cancer]", required=True)
    parser.add_argument("--kFold", type=int,
                        help="Use kFold > 0 to run KFold Cross-validation to get optimum model parameter this option overrides the default K,min_leaf option, default is 0 (no cv done)",
                        default=0)
    parser.add_argument("-p", default=[], nargs='+', type=int,
                        help="List of Cross-Validation parameter, this list should contain the model's parameters to be checked against the CV")
    parser.add_argument("-K", type=int, help="Set k parameter for the KNN experiments (default=9)", default=9)
    parser.add_argument("-M", type=int, help="Set min_leaf_samples parameter for ID3 experiments (default=100)",
                        default=100)
    vals = parser.parse_args()

    datasets = {"fifa19": Fifa19("fifa19", knn_k=vals.K, min_leaf_samples=vals.M),
                "cancer": Cancer("cancer", knn_k=vals.K, min_leaf_samples=vals.M)}
    if vals.exp not in [1, 2, 3, 4, 5] or vals.dataset not in datasets.keys():
        parser.print_help()
        return 0
    dataset = datasets[vals.dataset]
    is_knn = vals.exp in [3, 4]
    stochastic_model = vals.exp in [2, 4]
    classifier = KNN if is_knn else ID3
    classifier_param = dataset.getKNNParams(stochastic=stochastic_model,
                                            crossValidate=vals.kFold,
                                            k_to_check=vals.p) if is_knn else dataset.getID3Params(
        stochastic=stochastic_model, crossValidate=vals.kFold, m_to_check=vals.p)
    Utils.run_experiment(vals.exp, dataset.getDL(), classifier, classifier_param, frac=(2 / 3), normlize=is_knn,
                         dataset_name=dataset.getDatasetName())


#

if __name__ == "__main__":
    main()
    # csv = pd.read_csv("datasets/cancer/test1.csv",index_col=0)
    # print(csv)
    # csv.to_csv("datasets/cancer/test1.csv", index=True,header=True)
