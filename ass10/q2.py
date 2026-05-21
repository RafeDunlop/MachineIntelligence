import numpy as np

def estimate_mean_cov(X, ys):
    k = np.max(ys) + 1
    d = X.shape[1]

    means = np.zeros((k, d))
    covariances = np.zeros((k, d, d))

    for i in range(k):
        X_i = X[ys == i]  # all samples in class i

        means[i] = np.mean(X_i, axis=0)
        covariances[i] = np.cov(X_i, rowvar=False)

    return means, covariances

def test1():
    print()
    import numpy as np

    X = np.array([
        [1.0, 2.0], [1.5, 2.5], [1.3, 1.8],  # Class 0
        [3.0, 3.0], [3.5, 3.5], [3.2, 2.8]  # Class 1
    ])
    ys = np.array([0, 0, 0, 1, 1, 1])

    expected_means = np.array([
        [1.2667, 2.1000],
        [3.2333, 3.1000]
    ])

    expected_covariances = np.array([
        [[0.0633, 0.0550], [0.0550, 0.1300]],
        [[0.0633, 0.0700], [0.0700, 0.1300]]
    ])

    means, covariances = estimate_mean_cov(X, ys)
    print(np.allclose(means, expected_means, atol=1e-3))
    print(np.allclose(covariances, expected_covariances, atol=1e-3))

if __name__ == '__main__':
    test1()