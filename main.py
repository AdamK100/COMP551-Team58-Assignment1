import numpy as np
from helpers import cosine_similarity, evaluate_acc
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

np.random.shuffle(inputs.diabetes_clean_data)
dt_training_data = inputs.diabetes_clean_data[:576, :19]
dt_training_labels = inputs.diabetes_clean_data[:576, 19]
dt_validation_data = inputs.diabetes_clean_data[576:806, :19]
dt_validation_labels = inputs.diabetes_clean_data[576:806, 19]
dt_testing_data = inputs.diabetes_clean_data[806:, :19]
dt_testing_labels = inputs.diabetes_clean_data[806:, 19]

dt = models.DecisionTree()
dt.validate_depth(dt_training_data, dt_training_labels, dt_validation_data, dt_validation_labels)
print("Tree depth: " + str(dt.max_depth))
dt_predictions = dt.predict(dt_testing_data)
dt_accuracy = evaluate_acc(dt_testing_labels , dt_predictions)
print('Diabetes Decision Tree accuracy: ' + str(dt_accuracy))

#similarities = []
#for i in range(19):
#similarities.append(cosine_similarity(inputs.diabetes_clean_data[:,i],inputs.diabetes_clean_data[:,19]))
#print(similarities)
#print(np.mean(similarities))