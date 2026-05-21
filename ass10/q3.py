import numpy as np
from scipy.stats import multivariate_normal


def px_given_y(x, means, covariances):
    k = means.shape[0]

    likelihoods = np.zeros(k)

    for i in range(k):
        likelihoods[i] = multivariate_normal.pdf(
            x,
            mean=means[i],
            cov=covariances[i]
        )

    return likelihoods

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

    np.set_printoptions(precision=3)
    print(px_given_y(x, means, covariances))

if __name__ == '__main__':
    test1()