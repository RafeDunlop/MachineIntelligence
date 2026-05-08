import numpy as np

def linear_regression(xs, ys):
    bias = np.ones((xs.shape[0], 1))
    X = np.hstack((bias, xs))
    return np.linalg.inv(X.T @ X) @ X.T @ ys

def test1():
    xs = np.arange(5).reshape((-1, 1))
    ys = np.arange(1, 11, 2)
    print(linear_regression(xs, ys))

if __name__ == "__main__":
    print()
    test1()