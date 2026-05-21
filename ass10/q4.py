import numpy as np
from scipy.stats import multivariate_normal


def py_given_x(x, means, covariances, phis):
    k = means.shape[0]

    unnormalized = np.zeros(k)

    for i in range(k):
        likelihood = multivariate_normal.pdf(
            x,
            mean=means[i],
            cov=covariances[i]
        )
        unnormalized[i] = likelihood * phis[i]

    total = np.sum(unnormalized)

    return unnormalized / total

def test1():
    print()
    import numpy as np

    x = np.array([1.0, 2.0])
    means = np.array([
        [1.0, 2.0],
        [3.0, 3.0]
    ])
    covariances = np.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]]
    ])
    phis = np.array([0.6, 0.4])

    np.set_printoptions(precision=4)
    print(py_given_x(x, means, covariances, phis))

if __name__ == '__main__':
    test1()