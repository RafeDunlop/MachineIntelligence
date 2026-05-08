from q2 import ConfusionMatrix

def tpr(conf_mat):
    return conf_mat.TP / (conf_mat.TP + conf_mat.FN)

def fpr(conf_mat):
    return conf_mat.FP / (conf_mat.FP + conf_mat.TN)

def roc_non_dominated(classifiers):
    tprs, fprs = [[rate(c) for _, c in classifiers] for rate in (tpr, fpr)]
    def roc_not_dominated(index):
        for i in range(len(classifiers)):
            if i == index:
                continue
            if (tprs[i] >= tprs[index] and fprs[i] <= fprs[index]) and (tprs[i] > tprs[index] or fprs[i] < fprs[index]):
                return False
        return True
    compliant = filter(roc_not_dominated, range(len(classifiers)))
    return list(map(lambda i: classifiers[i], compliant))

def test1():
    print()
    # Example similar to the lecture notes

    classifiers = [
        ("h1", ConfusionMatrix(60, 40,
                               20, 80)),
        ("h2", ConfusionMatrix(40, 60,
                               30, 70)),
        ("h3", ConfusionMatrix(80, 20,
                               50, 50)),
    ]
    print(sorted(label for (label, _) in roc_non_dominated(classifiers)))

if __name__ == "__main__":
    test1()