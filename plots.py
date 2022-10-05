import matplotlib.pyplot as plt
import numpy as np

def plot(hyperparamters: np.ndarray, accuracies: np.ndarray, hyperparameter_name: str):
    plt.plot(hyperparamters, accuracies * 100)
    plt.xlabel(hyperparameter_name)
    plt.ylabel("Accuracy")
    plt.show()
