import numpy as np

def polynomial_kernel(degree):
    def k(x, z):
        s = np.dot(x, z)
        total = 0.0
        power = 1.0
        for _ in range(degree + 1):
            total += power
            power *= s
        return total
    return k

def test1():
    print()
    # The monomial kernel of order 1 is just fitting a bias
    k = polynomial_kernel(1)
    x = np.array([1])
    y = np.array([2])
    print(k(x, y) == 1 + (x * y).item())

def test2():
    print()
    import numpy as np
    # The feature map in the question example
    phi = lambda z: [1, z[0], z[1], z[0] * z[1], z[1] * z[0], z[0] ** 2, z[1] ** 2]
    k = polynomial_kernel(2)
    x = np.array([1, 2])
    y = np.array([3, 0.5])
    print(k(x, y) == np.dot(phi(x), phi(y)))


if __name__ == '__main__':
    test1()
    test2()

