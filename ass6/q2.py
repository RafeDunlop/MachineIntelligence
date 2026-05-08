import numpy as np
from collections import namedtuple

class ConfusionMatrix(namedtuple("ConfusionMatrix", "TP FN FP TN")):
    def __str__(self):
        return f"{self.TP:>3} {self.FN:>3}\n{self.FP:>3} {self.TN:>3}"

def confusion_matrix(classifier, dataset):
    C = np.zeros((2, 2), dtype=int)
    for params, classification in dataset:
        C[classification, classifier(params)] += 1
    return ConfusionMatrix(C[1, 1], C[1, 0], C[0, 1], C[0, 0])

def test1():
    print()
    dataset = [
        ((0.8, 0.2), 1),
        ((0.4, 0.3), 1),
        ((0.1, 0.35), 0),
    ]
    print(confusion_matrix(lambda x: 1, dataset))
    print()
    print(confusion_matrix(lambda x: 1 if x[0] + x[1] > 0.5 else 0, dataset))

if __name__ == "__main__":
    test1()