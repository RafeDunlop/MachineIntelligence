import numpy as np

def rbf_kernel(sigma):
    def k(x, y):
        diff = x - y
        sq_dist = np.dot(diff, diff)
        return np.exp(-sq_dist / (2 * sigma**2))
    return k

def test1():
    print()
    import numpy as np
    x = np.array([0.1])
    y = np.array([-0.25])
    k = rbf_kernel(1)
    print(f"RBF: {k(x, y):.6f}")

if __name__ == "__main__":
    test1()