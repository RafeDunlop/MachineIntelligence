import numpy as np


def one_hot_encoding(ys, classes=None):
    if not len(ys):
        return np.array([])
    k = max(ys) + 1 if classes is None else classes
    encoding = np.zeros((len(ys), k), dtype=int)
    for i in range(len(ys)):
        encoding[i, ys[i]] = 1
    return encoding

def test1():
    print()
    import numpy

    ys = numpy.array([0, 1, 0, 2, 1])
    print(one_hot_encoding(ys))

def test2():
    print()
    import numpy
    ys = numpy.array([])
    print(one_hot_encoding(ys))

def test3():
    print()
    import numpy
    ys = numpy.array([0])
    print(one_hot_encoding(ys))