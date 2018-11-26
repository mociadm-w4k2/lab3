import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prepare_data(data):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    features = data.iloc[:, 0:-1].values.astype(float)
    labels = data.iloc[:, -1].values.astype(str)
    classes = np.unique(labels)
    return features, labels, classes


def plot_results(train_time, scores, names, filename=None):
    plt.figure()
    plt.plot(train_time, scores, 'o')
    for t, s, n in zip(train_time, scores, names):
        plt.annotate(n, (t, s))
    plt.title(filename)
    plt.ylabel("Score")
    plt.xlabel("Time [s]")
    if filename is not None:
        plt.savefig("plots/%s.jpg" % filename)
    plt.show()
