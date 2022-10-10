import numpy as np
from helpers import entropy, evaluate_acc, misclass_rate, remove_irrelevant_features, manhattan_distance
import models
import inputs
from plots import plot

np.random.shuffle(inputs.hepatitis_clean_data)

knn = models.KNN_Graph()

#Using 50% training, 25% validation, 25% testing.

knn.fit(inputs.hepatitis_clean_data[:40], inputs.hepatitis_clean_data[:40, 0]) 
#validate_k takes as input the validation data, validation true labels, and the maximum value of k to test.
knn.validate_k(inputs.hepatitis_clean_data[41:61, 1:],inputs.hepatitis_clean_data[41:61, 0], 15)
print("Chosen K: " + str(knn.k))
predictions = knn.predict(inputs.hepatitis_clean_data[61:, 1:])

accuracy = evaluate_acc(inputs.hepatitis_clean_data[61:, 0], predictions)

k_trial_euclidean = knn.k_trial(inputs.hepatitis_clean_data[41:61, 1:],inputs.hepatitis_clean_data[41:61, 0], 15)
knn.dist_fn = manhattan_distance
k_trial_manhattan = knn.k_trial(inputs.hepatitis_clean_data[41:61, 1:],inputs.hepatitis_clean_data[41:61, 0], 15)

plot(range(1, 15), [k_trial_euclidean, k_trial_manhattan], "K", ["Euclidean", "Manhattan"])

print(f'Hepatitis KNN accuracy: {accuracy}')

# knn = models.KNN_Graph()

# knn.fit(inputs.diabetes_clean_data.shape[:550], inputs.diabetes_clean_data[:550, 0])
# predictions = knn.predict(inputs.diabetes_clean_data[551:, 1:])

# accuracy = evaluate_acc(inputs.diabetes_clean_data[551:, 0], predictions)

# print(f'Diabetes KNN accuracy: {accuracy}')

np.random.shuffle(inputs.diabetes_clean_data)

true_labels = inputs.diabetes_clean_data[:, -1]
processed_data = remove_irrelevant_features(inputs.diabetes_clean_data[:,:-1], true_labels, 4)

dt_training_data = processed_data[:576, :]
dt_training_labels = true_labels[:576]
dt_validation_data = processed_data[576:806, :]
dt_validation_labels = true_labels[576:806]
dt_testing_data = processed_data[806:, :]
dt_testing_labels = true_labels[806:]

dt = models.DecisionTree()
dt.validate_depth(dt_training_data, dt_training_labels, dt_validation_data, dt_validation_labels, 8)
print("Tree depth: " + str(dt.max_depth))

plot(range(1, 8), dt.max_depth_trial(dt_training_data, dt_training_labels, dt_validation_data, dt_validation_labels, 8), "Max depth")

dt.fit(dt_training_data, dt_training_labels)
dt_predictions = dt.predict(dt_testing_data)
dt_accuracy = evaluate_acc(dt_testing_labels , dt_predictions)
print('Diabetes Decision Tree accuracy: ' + str(dt_accuracy))

#similarities = []
#for i in range(processed_data.shape[1]):
    #similarities.append(cosine_similarity(processed_data[:,i],true_labels))
#print(similarities)
