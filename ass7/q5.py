import numpy as np
import random
import math
from collections import defaultdict

from scipy.constants import epsilon_0


def adaboost(learner, dataset, n_models):
    n = len(dataset)
    k = np.unique(dataset[:, -1])
    weights = np.ones(n) / n
    models, alphas = [], []

    def error(model):
        err = 0.
        predictions = []
        for i in range(n):
            x, y, = dataset[i, :-1], dataset[i, -1]
            y_hat_t = model(x)
            predictions.append(y_hat_t)
            if y_hat_t != y:
                err += weights[i]
        return err, predictions

    def update_weights():
        for i in range(n):
            if predictions[i] != dataset[i, -1]:
                continue
            if epsilon_t == 0:
                weights[i] = 0
            else:
                weights[i] *= (epsilon_t / (1 - epsilon_t))
        return weights / np.sum(weights)


    for _ in range(n_models):
        indices = random.choices(range(n), weights=weights, k=n)
        sample = dataset[indices]
        h_t = learner(sample)
        epsilon_t, predictions = error(h_t)
        print(epsilon_t)
        if epsilon_t >= 1 - (1/k):
            break
        try:
            alpha_t = math.log((1 - epsilon_t) / epsilon_t) + math.log(k - 1)
        except ZeroDivisionError:
            alpha_t = float('inf')

        weights = update_weights()
        models.append(h_t)
        alphas.append(alpha_t)

    def h_different_hat(x):
        scores = defaultdict(float)
        for h_t, alpha_t in zip(models, alphas):
            y_hat = h_t(x)
            scores[y_hat] += alpha_t
        return max(scores, key=scores.get)
    return h_different_hat

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
