from collections import Counter

def estimate_phis(ys):
    count = Counter(ys)
    total = count.total()
    return [count[label] / total for label in sorted(count.keys())]

def test1():
    print()
    import numpy as np

    ys = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    expected_output = np.array([1 / 3, 1 / 3, 1 / 3])

    your_output = estimate_phis(ys)

    print(np.allclose(your_output, expected_output))

def test2():
    print()
    from sklearn.datasets import load_wine
    data = load_wine()
    ys = data.target
    print(" ".join(f"{p:.4f}" for p in estimate_phis(ys)))

if __name__ == '__main__':
    test1()
    test2()