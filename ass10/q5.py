import numpy as np

def generate_data(n_samples, means, covariances, phis):
    k, d = means.shape

    data = np.zeros((n_samples, d + 1))

    for i in range(n_samples):
        y = np.random.choice(k, p=phis)

        x = np.random.multivariate_normal(
            mean=means[y],
            cov=covariances[y]
        )

        data[i, :d] = x
        data[i, d] = y

    return data

def test1():
    print()
    import numpy as np
    from scipy.stats import multivariate_normal, chisquare

    np.random.seed(0xC05C401)

    n_samples = 10000
    means = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    covariances = np.array([
        [[1.0, 0.2], [0.2, 1.0]],
        [[1.5, 0.3], [0.3, 1.5]]
    ])
    phis = np.array([0.7, 0.3])

    data = generate_data(n_samples, means, covariances, phis)

    assert isinstance(data, np.ndarray)
    assert data.shape == (n_samples, 3)

    # y distribution check
    _, y_counts = np.unique(data[:, -1], return_counts=True)
    expected_counts = n_samples * phis
    _, p_value = chisquare(y_counts, expected_counts)
    assert p_value > 0.01

    # x distribution check
    for i in range(len(means)):
        x_class = data[data[:, -1] == i, :-1]
        empirical_mean = np.mean(x_class, axis=0)
        empirical_cov = np.cov(x_class, rowvar=False)

        assert np.allclose(empirical_mean, means[i], atol=0.2)
        assert np.allclose(empirical_cov, covariances[i], atol=0.2)

    # reproducible?
    np.random.seed(0xC05C401)
    data2 = generate_data(n_samples, means, covariances, phis)
    assert np.allclose(data, data2)

    print("All tests passed.")

if __name__ == "__main__":
    test1()