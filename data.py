import numpy as np
import matplotlib.pyplot as plt

from utils import rad2degree
from matplotlib.pyplot import cm
from sklearn.model_selection import train_test_split

"""
- numpy trig functions expect angles to be in radians.
- we want to represent angles in the range (0, 2π) so we
    need to use both a cos and sin value to recover a unique
    point on the unit circle.
- atan2 returns angles in the range (-π, +π). We can % 2π to
    map to [0, 2π].
"""

def generate_data(size, dims=2, test_size=0.1, seed=0):
    np.random.seed(seed)
    X = np.random.randn(size, dims)
    c1 = np.random.randint(-20, 20, (dims,))
    c2 = np.random.randint(-20, 20, (dims,))
    sin = np.dot(X, c1)
    cos = np.dot(X, c2)
    y = np.arctan2(sin, cos).reshape(-1, 1)
    y = y % 2*np.pi
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    return train_test_split(X, y, test_size=test_size, random_state=seed)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data(500)

    fig, ax = plt.subplots(figsize=(8, 5))
    points = ax.scatter(
        X_train[:, 0], X_train[:, 1],
        c=rad2degree(y_train).squeeze(),
        cmap=cm.rainbow,
    )
    fig.colorbar(points)
    plt.show()
