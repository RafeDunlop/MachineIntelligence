import numpy as np
from q3 import polynomial_kernel

def kernel_linear_regression(X, y, k, alpha, iterations):
    n = X.shape[0]
    K = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = k(X[i], X[j])
    a = np.zeros(n)
    for _ in range(iterations):
        gradient = K @ a - y
        a = a - alpha * gradient
    def predictor(x):
        k_vec = np.array([k(X[i], x) for i in range(n)])
        return np.dot(a, k_vec)
    return predictor

def test1():
    print()
    import numpy as np

    def f(x):
        return x ** 2 - 1.0

    xs = np.arange(-3, 4, 0.5) * 0.5
    X = xs.reshape(-1, 1)
    y = f(xs)

    alpha = 0.02
    iterations = 1_000

    kernel = polynomial_kernel(degree=2)
    h = kernel_linear_regression(X, y, kernel, alpha, iterations)

    test_xs = np.array([1.2, -1, 0.7])
    X_test = test_xs.reshape(-1, 1)

    y_true = f(test_xs)
    y_pred = np.array([h(x) for x in X_test])

    print(np.allclose(y_pred, y_true, atol=0.01))

def test2():
    print()
    import numpy as np
    np.random.seed(0xc05c401)

    def f(x):
        return x[:, 0] ** 2 + x[:, 1] ** 2

    X = np.random.random((30, 2)) - 0.5
    y = f(X)

    alpha = 0.05
    iterations = 2000

    kernel = polynomial_kernel(degree=2)
    h = kernel_linear_regression(X, y, kernel, alpha, iterations)

    X_test = 3 * np.random.random((10, 2)) - 1.5

    y_true = f(X_test)
    y_pred = np.array([h(x) for x in X_test])

    print(np.allclose(y_pred, y_true, atol=0.001))

if __name__ == '__main__':
    test1()
    test2()