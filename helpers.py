import numpy as np

def euclidean_distance(x1: list[float], x2: list[float]) -> float:
    if len(x1) != len(x2): return -1

    dist_squared: float = 0
    for i in range(len(x1)):
        dist_squared += (x1[i] - x2[i]) ** 2
    
    return np.sqrt(dist_squared)


def most_common_label(l: list):
    map: dict = {}
    for item in l:
        if item in map.keys():
            map[item] = map[item] + 1
        else:
            map[item] = 1
            

    max_label = list(map.keys())[0]
    max: int = map[max_label]
    for key in map:
        if map[key] > max:
            max_label = key
            max = map[max_label]
    
    return max_label


def evaluate_acc(true_labels: list, target_labels: list) -> float:
    return np.sum(true_labels == target_labels) / len(true_labels)

def gini_index(lbls):
    cprobs = np.bincount(lbls.astype(int)) / len(lbls)
    gini = 0
    for p in cprobs:
        gini += (p * (1-p))
    return gini
