import numpy as np

def linear_regression(xs, ys, basis_functions=None, penalty=0.):
    if basis_functions is None:
        Phi = xs
    elif not len(basis_functions):
        Phi = np.empty((xs.shape[0], 0))
    else:
        Phi = np.column_stack([[f(x) for x in xs] for f in basis_functions])
    Phi = np.column_stack([np.ones(xs.shape[0]), Phi])
    XtX, XtY = Phi.T @ Phi, Phi.T @ ys
    R = penalty * np.eye(XtX.shape[0])
    R[0, 0] = 0
    return np.linalg.inv(XtX + R) @ XtY

def test1():
    print()
    import numpy as np

    xs = np.arange(5).reshape((-1, 1))
    ys = np.arange(1, 11, 2)

    print(linear_regression(xs, ys), end="\n\n")

    with np.printoptions(precision=5, suppress=True):
        print(linear_regression(xs, ys, penalty=0.1))

def test2():
    print()
    import numpy as np

    # we set the seed to some number so we can replicate the computation
    np.random.seed(0)

    xs = np.arange(-1, 1, 0.1).reshape(-1, 1)
    m, n = xs.shape
    # Some true function plus some noise:
    ys = (xs ** 2 - 3 * xs + 2 + np.random.normal(0, 0.5, (m, 1))).ravel()

    functions = [lambda x: x[0], lambda x: x[0] ** 2, lambda x: x[0] ** 3, lambda x: x[0] ** 4,
                 lambda x: x[0] ** 5, lambda x: x[0] ** 6, lambda x: x[0] ** 7, lambda x: x[0] ** 8]

    for penalty in [0, 0.01, 0.1, 1, 10]:
        with np.printoptions(precision=5, suppress=True):
            print(linear_regression(xs, ys, basis_functions=functions, penalty=penalty)
                  .reshape((-1, 1)), end="\n\n")

if __name__ == "__main__":
    test1()
    test2()


