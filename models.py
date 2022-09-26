from dis import dis
from typing import Callable
import numpy as np

from helpers import euclidean_distance, most_common_label

class Model:
    def __init__(self):
        return
    
    def fit(self, training_data, true_labels) -> "Model":
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

    def fit(self, training_data: np.array, true_labels: list[str]) -> "KNN_Graph":
        self.training_data = training_data
        self.true_labels = true_labels
        self.num_classes = np.max(true_labels) # TODO: Fix the way to detect number of classes
        return self
    
    def predict(self, test_data: np.array) -> list:
        distances: list[list[float]] = []

        for test_data_point in test_data:
            test_data_point_distances: list[float] = []
            for training_data_point in self.training_data:
                test_data_point_distances.append([self.dist_fn(test_data_point, training_data_point[1:]), training_data_point[0]])
            
            test_data_point_distances.sort()

            distances.append(test_data_point_distances)
        
        distances = np.array(distances)
        
        predictions: list = []
        for d in distances:
            predictions.append(most_common_label(d[:self.k, 1]))
        
        return predictions
        
        

class DecisionTree(Model):

    max_depth: int

    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth

    def fit(self, training_data: np.array, true_labels: list[str]) -> "DecisionTree":
        return
    
    def predict(self, input: np.array) -> list:
        return