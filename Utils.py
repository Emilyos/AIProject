import enum
from KNN import KNN
import ID3
from ForestClassifier import ForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


class FeatureType(enum.Enum):
    Discrete = 0
    Continuous = 1


class Feature:
    def __init__(self, index, type, domain=[], used=False):
        self.index = index
        self.type = type
        self.domain = domain
        self.used = used


def exp_5(dataloader, knn_params, id3_params, dataset_name="", frac=(2 / 3)):
    print(f"Running experiment 5 on {dataset_name} dataset")
    result_file_name = f"results/exp_5_{dataset_name}"
    performances = []
    for i in range(4, 22, 4):
        sub_forest_trees_number = i // 4
        knn_params['stochastic'] = False
        forest_knn = ForestClassifier(sub_forest_trees_number, dataloader, KNN, dict(knn_params), normlize=True)
        knn_params['stochastic'] = True
        forest_stochastic_knn = ForestClassifier(sub_forest_trees_number, dataloader, KNN, dict(knn_params),
                                                 normlize=True)
        id3_params['stochastic'] = False
        forest_id3 = ForestClassifier(sub_forest_trees_number, dataloader, ID3.ID3, dict(id3_params), normlize=False)
        id3_params['stochastic'] = True
        forest_stochastic_id3 = ForestClassifier(sub_forest_trees_number, dataloader, ID3.ID3, dict(id3_params),
                                                 normlize=False)

        forest_knn.buildForest(frac)
        forest_stochastic_knn.buildForest(frac)
        forest_id3.buildForest(frac)
        forest_stochastic_id3.buildForest(frac)

        results = np.empty(shape=(4, dataloader.n_test_samples))
        results[0] = forest_knn.performance()[2]
        results[1] = forest_stochastic_knn.performance()[2]
        results[2] = forest_id3.performance()[2]
        results[3] = forest_stochastic_id3.performance()[2]
        votes = np.empty(shape=dataloader.n_test_samples, dtype=int)
        for k in range(dataloader.n_test_samples):
            bin_count = np.bincount(results[:, k].astype(np.int))
            votes[k] = bin_count.argmax()
        accuracy = accuracy_score(dataloader.test_samples()[1], votes) * 100
        performances.append((i, accuracy))
    result_file = open(f"{result_file_name}.txt", 'w')
    for perf in performances:
        result_file.write(f"{perf[0]} -> {perf[1]}\n")
    plt.figure()
    plt.plot([ele[0] for ele in performances], [ele[1] for ele in performances])
    plt.title(f"Experiment 5 - {dataset_name}")
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    plt.savefig(f"{result_file_name}.png")


def run_experiment(exp_number, dataloader, classifier, classifier_params, frac=(2 / 3), normlize=False,
                   dataset_name=""):
    print(f"Running experiment {exp_number} on {dataset_name} dataset")
    result_file_name = f"results/exp_{exp_number}_{dataset_name}"

    performances = []

    for i in range(1, 22, 2):
        forest = ForestClassifier(i, dataloader, classifier, classifier_params, normlize=normlize)
        forest.buildForest(frac=frac)
        tree, accuracy, _ = forest.performance()
        performances.append((tree, accuracy))
    result_file = open(f"{result_file_name}.txt", 'w')
    for perf in performances:
        result_file.write(f"{perf[0]} -> {perf[1]}\n")
    plt.figure()
    plt.plot([ele[0] for ele in performances], [ele[1] for ele in performances])
    plt.title(f"Experiment {exp_number} - {dataset_name}")
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    plt.savefig(f"{result_file_name}.png")


def CrossValidateID3(k_fold, dataloader, m_to_check, stochastic=False, frac=1, normlize=False, shuffle=False,
                     dataset_name=""):
    X_train, y_train = dataloader.train_samples(normlize=normlize, frac=frac, shuffle=shuffle)
    ss = "[ "
    for i, k in enumerate(m_to_check):
        ss += f"{k}"
        if i != len(m_to_check) - 1:
            ss += ", "
    ss += " ]"
    stochastic_string = "_Stochastic" if stochastic else ""
    result_file_name = f"results/{k_fold}CV{stochastic_string}_ID3_{dataset_name}"
    best_m, accuracies = ID3.ID3.crossValidate(k_fold, X_train, y_train, m_to_check, dataloader.getFeatures(),
                                               stochastic=stochastic)

    result_file = open(f"{result_file_name}.txt", 'w')
    stochastic_string = "Stochastic" if stochastic else ""
    result_file.write(f"{k_fold}-Cross Validation on {stochastic_string} ID3 model\n")
    result_file.write(f"\tDataset\t{dataset_name}\n")
    result_file.write(f"\tM Checked:\t{ss}\n")
    result_file.write(f"\tNormlize Feature:\t{normlize}\n")
    result_file.write(f"-------------------------\n")
    for k, acc in accuracies:
        result_file.write(f"Model with m={k} got an accuracy of {acc}%\n\n")
    result_file.write(f"-------------------------\nBest m: {best_m}")
    plt.figure()
    plt.plot([ele[0] for ele in accuracies], [ele[1] for ele in accuracies])
    plt.title(f"{k_fold}-Fold CV {stochastic_string} ID3 {dataset_name}")
    plt.xlabel("min_leaf_samples")
    plt.ylabel("Accuracy")
    plt.savefig(f"{result_file_name}.png")
    return best_m


def CrossValidateKNN(k_fold, dataloader, k_to_check, stochastic=False, frac=1, normlize=True, shuffle=False,
                     dataset_name=""):
    X_train, y_train = dataloader.train_samples(normlize=normlize, frac=frac, shuffle=shuffle)
    ss = "[ "
    for i, k in enumerate(k_to_check):
        ss += f"{k}"
        if i != len(k_to_check) - 1:
            ss += ", "
    ss += " ]"
    stochastic_string = "_Stochastic" if stochastic else ""
    result_file_name = f"results/{k_fold}CV{stochastic_string}_KNN_{dataset_name}"
    best_k, accuracies = KNN.crossValidate(k_fold, X_train, y_train, k_to_check, stochastic,
                                           dist_func=dataloader.dist_func)

    result_file = open(f"{result_file_name}.txt", 'w')
    stochastic_string = "Stochastic" if stochastic else ""
    result_file.write(f"{k_fold}-Cross Validation on {stochastic_string} KNN model\n")
    result_file.write(f"\tDataset\t{dataset_name}\n")
    result_file.write(f"\tK Checked:\t{ss}\n")
    result_file.write(f"\tNormlize Feature:\t{normlize}\n")
    result_file.write(f"-------------------------\n")
    for k, acc in accuracies:
        result_file.write(f"Model with k={k} got an accuracy of {acc}%\n\n")

    result_file.write(f"-------------------------\nBest k: {best_k}")
    plt.figure()
    plt.plot([ele[0] for ele in accuracies], [ele[1] for ele in accuracies])
    plt.title(f"{k_fold}-Fold CV {stochastic_string} KNN {dataset_name}")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.savefig(f"{result_file_name}.png")
    return best_k
