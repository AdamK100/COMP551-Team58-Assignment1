from typing import Callable
import numpy as np

from helpers import euclidean_distance

class Model:
    def __init__(self):
        return
    
    def fit(self, hyperparameter: int, training_data, true_labels) -> "Model":
        return
    
    def predict(self, input):
        return


class KNN_Graph(Model):

    k: int
    dist_fn: Callable[[list[float], list[float]], float]
    training_data: np.array
    true_labels: list[str]
    num_classes: int

    def __init__(self, k: int = 1, dist_fn: Callable[[list[float], list[float]], float] = euclidean_distance):
        self.k = k
        self.dist_fn = dist_fn

    def fit(self, k: int, training_data: np.array, true_labels: list[str]) -> "KNN_Graph":
        self.training_data = training_data
        self.true_labels = true_labels
        self.num_classes = np.max(true_labels) # TODO: Fix the way to detect number of classes
        return self
    
    def predict(self, test_data: np.array) -> list:
        num_tests = test_data.shape[0]
        distances = []
        
        print(distances)
        

class DecisionTree(Model):

    max_depth: int

    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth

    def fit(self, max_depth: int, training_data: np.array, true_labels: list[str]) -> "DecisionTree":
        return
    
    def predict(self, input: np.array) -> list:
        return