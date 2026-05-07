from functools import reduce

def reducer(counts, classification):
    if classification in counts:
        counts[classification] += 1
    else:
        counts[classification] = 1
    return counts

def voting_ensemble(classifiers):
    def classify(x):
        votes = reduce(lambda counts, classifier: reducer(counts, classifier(x)), classifiers, {})
        high_score = votes[max(votes, key=votes.get)]
        return min([classification for classification in votes.keys() if votes[classification] == high_score])
    return classify

def test1():
    print()
    # Modelling y > x^2
    classifiers = [
        lambda p: 1 if 1.0 * p[0] < p[1] else 0,
        lambda p: 1 if 0.9 * p[0] < p[1] else 0,
        lambda p: 1 if 0.8 * p[0] < p[1] else 0,
        lambda p: 1 if 0.7 * p[0] < p[1] else 0,
        lambda p: 1 if 0.5 * p[0] < p[1] else 0,
    ]
    data_points = [(0.2, 0.03), (0.1, 0.12),
                   (0.8, 0.63), (0.9, 0.82)]
    ve = voting_ensemble(classifiers)
    for x in data_points:
        print(ve(x))

def test2():
    print()
    import numpy as np
    import random
    random.seed(0)
    np.random.seed(0)

    from sklearn import datasets, utils
    from sklearn import tree, svm, neighbors

    iris = datasets.load_iris()
    data, target = utils.shuffle(iris.data, iris.target, random_state=1)
    train_data, train_target = data[:-10, :], target[:-10]
    test_data, test_target = data[-10:, :], target[-10:]

    models = [
        tree.DecisionTreeClassifier(random_state=1),
        svm.SVC(random_state=1),
        neighbors.KNeighborsClassifier(3),
    ]

    for model in models:
        model.fit(train_data, train_target)

    def make_classifier(model):
        "A simple wrapper to adapt model to the type we need."
        return lambda x: model.predict(x)[0]

    classifiers = [make_classifier(model) for model in models]

    ve_classifier = voting_ensemble(classifiers)

    print("h1   h2   h3   ve   y")
    for (x, y) in zip(test_data, test_target):
        x = np.array([x])
        for h in classifiers:
            print(h(x), end="    ")
        print(ve_classifier(x), end="    ")
        print(y)  # ground truth

if __name__ == '__main__':
    test1()
    test2()