import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import rad2degree
from matplotlib.pyplot import cm
from sklearn.model_selection import train_test_split


def generate_data(size, dims=2, test_size=0.1, seed=0, show=False):
    np.random.seed(seed)

    # sample from standard normal
    X = np.random.randn(size, dims)

    # create random coefficients sampled from [-10, 10]
    c1 = np.random.randint(-10, 10, (dims,))
    c2 = np.random.randint(-10, 10, (dims,))

    # using coefficients, create sin and cos
    # values using a linear combination of the
    # columns of X.
    # note that these values can be outside
    # the range [0, 1] but since arctan2 takes
    # the ratio of sin and cosine, the support
    # is from [-inf, +inf].
    sin = np.dot(X, c1)
    cos = np.dot(X, c2)

    # create angle values. arctan2 returns them in the range [-π, +π]
    y = np.arctan2(sin, cos).reshape(-1, 1)

    # cast to float and return
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    if show:
        fig, ax = plt.subplots(figsize=(8, 5))
        points = ax.scatter(
            X[:, 0], X[:, 1],
            c=rad2degree(y).squeeze(),
            cmap=cm.Spectral.reversed(),
        )
        fig.colorbar(points)
        plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    return (X_train, X_test, y_train, y_test)
