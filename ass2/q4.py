from helpers import gini, misclassification, entropy

def test_all():
    data = [
        ((False, False), False),
        ((False, True), True),
        ((True, False), True),
        ((True, True), False)
    ]
    print("{:.4f}".format(misclassification(data)))
    print("{:.4f}".format(gini(data)))
    print("{:.4f}".format(entropy(data)))


if __name__ == "__main__":
    test_all()
