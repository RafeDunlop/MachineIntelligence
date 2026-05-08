import numpy as np

def linear_regression(xs, ys, basis_functions=None):
    if basis_functions is None:
        Phi = xs
    elif not len(basis_functions):
        Phi = np.empty((xs.shape[0], 0))
    else:
        Phi = np.column_stack([[f(x) for x in xs] for f in basis_functions])
    Phi = np.column_stack([np.ones(xs.shape[0]), Phi])
    XtX, XtY = Phi.T @ Phi, Phi.T @ ys
    return np.linalg.inv(XtX) @ XtY

def test1():
    print()
    import numpy as np

    xs = np.arange(5).reshape((-1, 1))
    ys = np.array([3, 6, 11, 18, 27])
    # Can you see y as a function of x? [hint: it's quadratic.]
    functions = [lambda x: x[0], lambda x: x[0] ** 2]
    print(linear_regression(xs, ys, functions))

def test3():
    print()
    import numpy as np

    xs = np.array([[1, 2, 3, 4],
                   [6, 2, 9, 1]]).T
    ys = np.array([7, 5, 14, 8])
    print(linear_regression(xs, ys, []) == np.average(ys))


if __name__ == '__main__':
    test1()


