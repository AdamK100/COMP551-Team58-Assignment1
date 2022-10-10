from math import log
import numpy as np
from math import sqrt

def euclidean_distance(x1: list[float], x2: list[float]) -> float:
    return np.sqrt(np.sum((np.array(x2) - np.array(x1)) ** 2))

def manhattan_distance(x1: list[float], x2: list[float]) -> float:
    return np.sum(np.abs(np.array(x2) - np.array(x1)))

def most_common_label(l: list):
    return np.bincount(np.array(l, int)).argmax()


def evaluate_acc(true_labels: list, target_labels: list) -> float:
    return np.sum(true_labels == target_labels) / len(true_labels)


def gini_index(labels):
    class_probs = np.bincount(labels.astype(int)) / len(labels)
    return 1 - np.sum(class_probs ** 2)

def misclass_rate(labels):
    return np.sum(labels != np.array([most_common_label(labels)] * len(labels)))/len(labels)

def entropy(labels):
    c_probs = np.bincount(labels.astype(int))/len(labels)
    s = 0
    for c in c_probs:
        if (c != 0): 
            s -= c * log(c,2)
    return s
    
def cosine_similarity(vector1,vector2):
    return np.sum(vector1 * vector2)/(sqrt(np.sum(vector1 ** 2)) * sqrt(np.sum(vector2 ** 2)))

def remove_irrelevant_features(data : np.ndarray, true_labels : np.ndarray, nb_features : int) -> np.ndarray:
    # similarities = np.apply_along_axis(lambda d: cosine_similarity(d, true_labels), 0, data)
    similarities = []
    for i in range(data.shape[1]):
        similarities.append(cosine_similarity(data[:,i],true_labels))
    min_similarities = np.argpartition(similarities,nb_features)
    return np.delete(data, min_similarities[:nb_features], 1)