import matplotlib.pyplot as plt
import numpy as np

colors = ['b', 'r', 'g']

def plot(x_data: np.ndarray, y_datas: list[np.ndarray], x_name: str, y_names: list[str]):
    figure = plt.figure()
    axes = figure.add_subplot(111)
    
    for i in range(len(y_datas)):
        axes.plot(x_data, y_datas[i] * 100, label = y_names[i], c=colors[i % 3])

    plt.xlabel(x_name)
    plt.ylabel("Accuracy")
    plt.axis([x_data[0], x_data[-1], 0, 100])
    
    plt.legend(loc='lower right')
    plt.show()
