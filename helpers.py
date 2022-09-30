import numpy as np
from math import sqrt

def euclidean_distance(x1: list[float], x2: list[float]) -> float:
    return np.sqrt(np.sum((np.array(x2) - np.array(x1)) ** 2))


def most_common_label(l: list):
    return np.bincount(np.array(l, int)).argmax()


def evaluate_acc(true_labels: list, target_labels: list) -> float:
    return np.sum(true_labels == target_labels) / len(true_labels)


def gini_index(labels):
    class_probs = np.bincount(labels.astype(int)) / len(labels)
    return 1 - np.sum(class_probs ** 2)

def cosine_similarity(vector1,vector2):
    return np.sum(vector1 * vector2)/(sqrt(np.sum(vector1 ** 2)) * sqrt(np.sum(vector2 ** 2)))

def remove_irrelevant_features(data : np.ndarray, true_labels : np.array, nb_features : int) -> np.ndarray:
    similarities = []
    for i in range(data.shape[1]):
        similarities.append(cosine_similarity(data[:,i],true_labels))
    min_similarities = np.argpartition(similarities,nb_features)
    return np.delete(data, min_similarities[:nb_features], 1)