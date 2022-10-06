import matplotlib.pyplot as plt
import numpy as np

def plot(hyperparameters: np.ndarray, accuracies: np.ndarray, hyperparameter_name: str):
    plt.plot(hyperparameters, accuracies * 100)
    plt.xlabel(hyperparameter_name)
    plt.ylabel("Accuracy")
    plt.axis([hyperparameters[0], hyperparameters[-1], 0, 100])
    plt.show()
