import numpy as np
from helpers import evaluate_acc

import models
import inputs

knn = models.KNN_Graph()

knn.fit(inputs.hepatitis_clean_data[:40], inputs.hepatitis_clean_data[:40, 0])
predictions = knn.predict(inputs.hepatitis_clean_data[41:, 1:])

accuracy = evaluate_acc(inputs.hepatitis_clean_data[41:, 0], predictions)

print(f'Hepaptitis KNN accuracy: {accuracy}')


# knn = models.KNN_Graph()

# knn.fit(inputs.diabetes_clean_data.shape[:550], inputs.diabetes_clean_data[:550, 0])
# predictions = knn.predict(inputs.diabetes_clean_data[551:, 1:])

# accuracy = evaluate_acc(inputs.diabetes_clean_data[551:, 0], predictions)

# print(f'Diabetes KNN accuracy: {accuracy}')
