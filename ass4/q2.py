import numpy as np

def softmax(z):
    normal = sum(np.exp(z))
    return np.exp(z) / normal

def test1():
    print()
    import numpy
    numpy.set_printoptions(precision=3, suppress=True)

    z = numpy.array([1, -1])
    print(softmax(z))
