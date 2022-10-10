from typing import Callable
import numpy as np

from helpers import euclidean_distance, evaluate_acc, most_common_label, gini_index

class Model:
    
    def __init__(self, hyperparameter: int):
        pass
    
    def fit(self, training_data, true_labels) -> "Model":
        pass
    
    def predict(self, input) -> list:
        pass


class KNN_Graph(Model):

    k: int
    dist_fn: Callable[[list[float], list[float]], float]
    training_data: np.ndarray
    true_labels: list[str]

    def __init__(self, k: int = 1, dist_fn: Callable[[list[float], list[float]], float] = euclidean_distance):
        self.distances_valid = False
        self.k = k
        self.dist_fn = dist_fn


    def fit(self, training_data: np.ndarray, true_labels: list[str]) -> "KNN_Graph":
        self.distances_valid = False
        self.training_data = training_data
        self.true_labels = true_labels
        return self
    

    def compute_distances_single_point(self, test_data_point):
        distances = np.apply_along_axis(lambda tdp: [self.dist_fn(test_data_point, tdp[1:]), tdp[0]], 1, self.training_data)
        distances = distances[distances[:, 0].argsort()] # Sort based row by first column
        return distances


    def compute_distances(self, test_data) -> np.ndarray:
        return np.apply_along_axis(self.compute_distances_single_point, 1, test_data)
    

    def predict(self, test_data: np.ndarray) -> list:
        distances = self.compute_distances(test_data)
        return [most_common_label(d[:self.k, 1]) for d in distances]
    

    def k_trial(self, validation_data, validation_labels, max_k: int) -> np.ndarray:
        distances = self.compute_distances(validation_data)

        accuracies = []
        for k in range(1, max_k):
            self.k = k
            predictions = [most_common_label(d[:k, 1]) for d in distances]
            accuracies.append(evaluate_acc(validation_labels,predictions))

        return np.array(accuracies)
        

    def validate_k(self, validation_data, validation_labels, max_k) -> "KNN_Graph":
        accuracies = self.k_trial(validation_data, validation_labels, max_k)
        self.k = np.argmax(accuracies) + 1
        return self

        
class Node:

    depth: int
    feature : int
    threshold : float
    
    def __init__(self, data_indices, parent):
        self.data_indices = data_indices
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
            self.data = parent.data
            self.labels = parent.labels   

                   

class DecisionTree(Model):

    max_depth: int
    root: Node
    cost_fn: Callable[[list[float]], float]
    def __init__(self, max_depth: int = 1, cost_fn: Callable[[list[float]], float] = gini_index):
        self.max_depth = max_depth
        self.cost_fn = cost_fn
        self.root = None


    def greedy_split(self, node):
        best_feature, best_threshold = None, None
        best_cost = np.inf
        nb_points, nb_features = node.data.shape
        sorted_data = np.sort(node.data[node.data_indices], axis=0)
        thresholds = test_candidates = (sorted_data[1:] + sorted_data[:-1]) / 2.
        for i in range(nb_features):
            feature_data = node.data[node.data_indices, i]
            for t in thresholds[:,i]:
                left_indices = node.data_indices[feature_data <= t]
                right_indices = node.data_indices[feature_data > t]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                left_cost = self.cost_fn(node.labels[left_indices])
                right_cost = self.cost_fn(node.labels[right_indices])
                num_left, num_right = left_indices.shape[0], right_indices.shape[0]
                cost = (num_left * left_cost + num_right * right_cost) / nb_points
                if cost < best_cost:
                    best_cost = cost
                    best_feature = i
                    best_threshold = t
        return best_cost, best_feature, best_threshold


    def fit(self, training_data: np.ndarray, true_labels: list[str]) -> "DecisionTree":
        root = Node(np.arange(len(training_data)), None)
        root.labels = true_labels
        root.data = training_data
        self.root = root
        self._fit_tree(self.root)
        return self
    

    def _fit_tree(self, node):
        if node.depth == self.max_depth or len(node.data_indices) <= 1:
            return
        cost, feature, threshold = self.greedy_split(node)
        if np.isinf(cost):
            return
        test = node.data[node.data_indices,feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        left = Node(node.data_indices[test], node)
        right = Node(node.data_indices[np.logical_not(test)], node)
        self._fit_tree(left)
        self._fit_tree(right)
        node.left = left
        node.right = right
        

    def predict(self, input: np.ndarray) -> list:
        predictions = []
        for i in range(input.shape[0]):
            predictions.append(self._classify(input[i], self.root, 0))
        return predictions


    def _classify(self, point, node, depth):
        if(node.threshold != None and node.feature != None and depth < self.max_depth):
            if(point[node.feature] <= node.threshold):
                return self._classify(point, node.left, depth+1)
            else:
                return self._classify(point, node.right, depth+1)
        else:
            return most_common_label(node.labels[node.data_indices])


    def max_depth_trial(self, input: np.ndarray, labels, validation_input, validation_labels, max_depth: int) -> np.ndarray:
        accuracies = []
        self.max_depth = max_depth
        self.fit(input, labels)
        for d in range(1, max_depth):
            self.max_depth = d
            predictions = self.predict(validation_input)
            accuracies.append(evaluate_acc(validation_labels, predictions))
        
        return np.array(accuracies)


    def validate_depth(self, input: np.ndarray, labels, validation_input, validation_labels, max_depth: int) -> "DecisionTree":
        accuracies = self.max_depth_trial(input, labels, validation_input, validation_labels, max_depth)
        self.max_depth = np.argmax(accuracies) + 1
        return self
