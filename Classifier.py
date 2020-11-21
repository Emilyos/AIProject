from abc import ABC, abstractmethod
import numpy as np


class Classifer(ABC):

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        pass

    @abstractmethod
    def predict(self, samples: np.ndarray):
        pass

    @abstractmethod
    def getClassifierName(self):
        pass
