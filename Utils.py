import enum
from KNN import KNN
from ForestClassifier import ForestClassifier
import matplotlib.pyplot as plt


class FeatureType(enum.Enum):
    Discrete = 0
    Continuous = 1


class Feature:
    def __init__(self, index, type, domain=[], used=False):
        self.index = index
        self.type = type
        self.domain = domain
        self.used = used


def run_experiment(exp_number, dataloader, classifier, classifier_params, frac=(2 / 3), normlize=False):
    print(f"Running experiment {exp_number} on {dataloader.datasetName()} dataset")
    result_file_name = f"results/exp_{exp_number}_{dataloader.datasetName()}"

    performances = []

    for i in range(1, 22, 2):
        forest = ForestClassifier(i, dataloader, classifier, classifier_params, normlize=normlize)
        forest.buildForest(frac=frac)
        performances.append((forest.performance()))
    result_file = open(f"{result_file_name}.txt", 'w')
    for perf in performances:
        result_file.write(f"{perf[0]} -> {perf[1]}\n")
    plt.figure()
    plt.plot([ele[0] for ele in performances], [ele[1] for ele in performances])
    plt.title(f"Experiment {exp_number} - {dataloader.datasetName()}")
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    plt.savefig(f"{result_file_name}.png")


def CrossValidateKNN(k_fold, dataloader, k_to_check, stochastic=False, frac=1, normlize=True, shuffle=False):
    X_train, y_train = dataloader.train_samples(normlize=normlize, frac=frac, shuffle=shuffle)
    X_test, y_test = dataloader.test_samples(normlize=normlize)
    ss = "[ "
    for i, k in enumerate(k_to_check):
        ss += f"{k}"
        if i != len(k_to_check) - 1:
            ss += ", "
    ss += " ]"
    result_file_name = f"results/{k_fold}CV_{dataloader.datasetName()}"
    best_k, accuracies = KNN.crossValidate(k_fold, X_train, y_train, k_to_check, stochastic)

    result_file = open(f"{result_file_name}.txt", 'w')
    stochastic_string = "Stochastic" if stochastic else " "
    result_file.write(f"{k_fold}-Cross Validation on{stochastic_string}KNN model\n")
    result_file.write(f"\tDataset\t{dataloader.datasetName()}\n")
    result_file.write(f"\tK Checked:\t{ss}\n")
    result_file.write(f"\tNormlize Feature:\t{normlize}\n")
    result_file.write(f"-------------------------\n")
    for k, acc in accuracies:
        result_file.write(f"Model with k={k} got an accuracy of {acc}%\n\n")

    result_file.write(f"-------------------------\nBest k: {best_k}")
    plt.figure()
    plt.plot([ele[0] for ele in accuracies], [ele[1] for ele in accuracies])
    plt.title(f"{k_fold}-Fold CV {stochastic_string} KNN {dataloader.datasetName()}")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.savefig(f"{result_file_name}.png")
    return best_k
