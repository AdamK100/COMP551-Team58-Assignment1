from typing import Callable
import numpy as np

from helpers import euclidean_distance, most_common_label, gini_index

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
        
class Node:
    depth: int
    feature : int
    threshold : float
    def __init__(self, data_indices, parent):
        self.data_indices = data_indices                   #stores the data indices which are in the region defined by this node
        self.left = None                                    #stores the left child of the node 
        self.right = None                                   #stores the right child of the node
        self.feature = None                           #the feature for split at this node
        self.threshold = None
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
            self.data = parent.data
            self.labels = parent.labels   

                   

class DecisionTree(Model):

    max_depth: int
    root: Node

    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth
        self.root = None  

    def greedy_split(self, node):
            best_feature, best_threshold = None, None
            best_cost = np.inf
            nb_points , nb_features = node.data.shape
            sorted_data = np.sort(node.data[node.data_indices], axis=0)
            thresholds = test_candidates = (sorted_data[1:] + sorted_data[:-1]) / 2.
            for i in range(nb_features):
                feature_data = node.data[node.data_indices, i]
                for t in thresholds[:,i]:
                    left_indices = node.data_indices[feature_data <= t]
                    right_indices = node.data_indices[feature_data > t]
                    if len(left_indices) == 0 or len(right_indices) == 0:                
                        continue                                                      
                    left_cost = gini_index(node.labels[left_indices])
                    right_cost = gini_index(node.labels[right_indices])
                    num_left, num_right = left_indices.shape[0], right_indices.shape[0]
                    cost = (num_left * left_cost + num_right * right_cost)/nb_points
                    if cost < best_cost:
                        best_cost = cost
                        best_feature = i
                        best_threshold = t
            return best_cost, best_feature, best_threshold

    def fit(self, training_data: np.array, true_labels: list[str]) -> "DecisionTree":
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
            predictions.append(self._classify(input[i] , self.root))
        return predictions

    def _classify(self, point, node):
        if(node.threshold != None and node.feature != None):
            if(point[node.feature] <= node.threshold):
                return self._classify(point, node.left)
            else:
                return self._classify(point, node.right)
        else:
            return most_common_label(node.labels[node.data_indices])
