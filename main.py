import numpy as np

import models
import inputs

knn = models.KNN_Graph()

knn.fit(1, inputs.hepatitis_clean_data, inputs.hepatitis_clean_data[0,:])

def evaluate_acc(true_labels: list, target_labels: list):
    return
