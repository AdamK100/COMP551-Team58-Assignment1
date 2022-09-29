import numpy as np
from helpers import cosine_similarity, evaluate_acc
import models
import inputs

np.random.shuffle(inputs.hepatitis_clean_data)

knn = models.KNN_Graph()

#Using 50% training, 25% validation, 25% testing.

knn.fit(inputs.hepatitis_clean_data[:40], inputs.hepatitis_clean_data[:40, 0]) 
#validate_k takes as input the validation data, validation true labels, and the maximum value of k to test.
knn.validate_k(inputs.hepatitis_clean_data[41:61, 1:],inputs.hepatitis_clean_data[41:61, 0], 8)
print("Chosen K: " + str(knn.k))
predictions = knn.predict(inputs.hepatitis_clean_data[61:, 1:])

accuracy = evaluate_acc(inputs.hepatitis_clean_data[61:, 0], predictions)

print(f'Hepatitis KNN accuracy: {accuracy}')

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
dt.validate_depth(dt_training_data, dt_training_labels, dt_validation_data, dt_validation_labels, 8)
print("Tree depth: " + str(dt.max_depth))
dt.fit(dt_training_data, dt_training_labels)
dt_predictions = dt.predict(dt_testing_data)
dt_accuracy = evaluate_acc(dt_testing_labels , dt_predictions)
print('Diabetes Decision Tree accuracy: ' + str(dt_accuracy))

#similarities = []
#for i in range(19):
#similarities.append(cosine_similarity(inputs.diabetes_clean_data[:,i],inputs.diabetes_clean_data[:,19]))
#print(similarities)
#print(np.mean(similarities))