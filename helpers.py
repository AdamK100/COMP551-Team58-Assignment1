import numpy as np

def euclidean_distance(x1: list[float], x2: list[float]) -> float:
    if len(x1) != len(x2): return -1

    dist_squared: float = 0
    for i in range(len(x1)):
        dist_squared = (x1[i] - x2[i]) ** 2
    
    return np.sqrt(dist_squared)


def evaluate_acc(true_labels: list, target_labels: list):
    return