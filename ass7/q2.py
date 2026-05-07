import random

def bootstrap(dataset, sample_size, seed=0):
    random.seed(seed)
    n = len(dataset)

    while True:
        indices = random.choices(range(n), k=sample_size)
        yield dataset[indices]

def test1():
    print()
    import numpy as np

    dataset = np.array([[1, 0, 2, 3],
                        [2, 3, 0, 0],
                        [4, 1, 2, 0],
                        [3, 2, 1, 0]])
    ds_gen = bootstrap(dataset, 3)

    for _ in range(5):
        print(next(ds_gen), end="\n\n")

    ds = next(ds_gen)
    print(type(ds))
    print(ds.dtype != object)

if __name__ == "__main__":
    test1()
