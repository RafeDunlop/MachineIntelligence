import numpy as np
import random
import math
from collections import defaultdict


def adaboost(learner, dataset, n_models):
    n = len(dataset)

    # number of classes
    classes = np.unique(dataset[:, -1])
    k = len(classes)

    # Step 1: initialise weights
    weights = np.ones(n) / n

    models = []
    alphas = []

    for _ in range(n_models):
        # Step 2(a): train using weighted bootstrap
        indices = random.choices(range(n), weights=weights, k=n)
        sample = dataset[indices]
        h_t = learner(sample)

        # Step 2(b): compute weighted error on FULL dataset
        epsilon_t = 0.0
        predictions = []

        for i in range(n):
            x = dataset[i, :-1]
            y = dataset[i, -1]
            pred = h_t(x)
            predictions.append(pred)
            if pred != y:
                epsilon_t += weights[i]

        # Step 2(c): early termination (multi-class version)
        if epsilon_t >= 1 - (1 / k):
            break

        # Step 2(d): compute alpha
        if epsilon_t == 0:
            alpha_t = float('inf')
        else:
            alpha_t = math.log((1 - epsilon_t) / epsilon_t) + math.log(k - 1)

        # Step 2(e): update weights
        for i in range(n):
            if predictions[i] == dataset[i, -1]:  # correctly classified
                if epsilon_t == 0:
                    weights[i] = 0
                else:
                    weights[i] *= (epsilon_t / (1 - epsilon_t))
            # incorrect ones unchanged

        # Step 2(f): renormalise
        weights = weights / np.sum(weights)

        models.append(h_t)
        alphas.append(alpha_t)

    # Step 3: final classifier (multi-class weighted vote)
    def H(x):
        scores = defaultdict(float)
        for h_t, alpha_t in zip(models, alphas):
            pred = h_t(x)
            scores[pred] += alpha_t
        return max(scores, key=scores.get)

    return H

def test1():
    print()
    import sklearn.datasets
    import sklearn.utils
    import sklearn.linear_model

    digits = sklearn.datasets.load_digits()
    data, target = sklearn.utils.shuffle(digits.data, digits.target, random_state=3)
    train_data, train_target = data[:-5, :], target[:-5]
    test_data, test_target = data[-5:, :], target[-5:]
    dataset = np.hstack((train_data, train_target.reshape((-1, 1))))

    def linear_learner(dataset):
        features, target = dataset[:, :-1], dataset[:, -1]
        model = sklearn.linear_model.SGDClassifier(random_state=1, max_iter=1000, tol=0.001).fit(features, target)
        return lambda v: model.predict(np.array([v]))[0]

    boosted = adaboost(linear_learner, dataset, 10)
    for (v, c) in zip(test_data, test_target):
        print(int(boosted(v)), c)

if __name__ == '__main__':
    test1()