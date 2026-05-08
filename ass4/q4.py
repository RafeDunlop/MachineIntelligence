import numpy as np
from q1 import aug
from q2 import softmax
from q3 import one_hot_encoding

mat_aug = lambda xs: np.concatenate((np.ones((xs.shape[0], 1)), xs), axis =1)

def grad_vector(xs, ys, theta):
    # had to adjust impl. of one-hot-encoding and add several 'catches' to do this in matrix form. Wasn't worth it.
    xs = mat_aug(xs)
    scores, ys = xs @ theta.T, one_hot_encoding(ys, len(ys))
    soft_maxed = np.array([softmax(score) for score in scores])
    return (xs.T @ (soft_maxed - ys)).T

def softmax_regression(xs, ys, learning_rate, num_iterations):
    theta = np.zeros((xs.shape[0], xs.shape[1] + 1))
    for _ in range(num_iterations):
        theta -= learning_rate * grad_vector(xs, ys, theta)
    return lambda x: np.argmax(theta @ aug(np.atleast_1d(x)))

def test1():
    print()
    # Toy multiclass problem where x is class 0 if 0 <= x < 1,
    # class 1 if 1 <= x < 2 and class 2 if 2 <= x < 3

    import numpy as np

    training_data = np.array([
        (0.17, 0),
        (0.79, 0),
        (2.66, 2),
        (2.81, 2),
        (1.58, 1),
        (1.86, 1),
        (2.97, 2),
        (2.70, 2),
        (1.64, 1),
        (1.68, 1)
    ])

    xs = training_data[:, 0].reshape((-1, 1))  # a 2D n-by-1 array
    ys = training_data[:, 1].astype(int)  # a 1D array of length n

    h = softmax_regression(xs, ys, 0.05, 750)

    test_inputs = [(1.30, 1), (2.25, 2), (0.97, 0), (1.07, 1), (1.51, 1)]
    print(f"{'prediction':^10}{'true':^10}")
    for x, y in test_inputs:
        print(f"{h(x):^10}{y:^10}")