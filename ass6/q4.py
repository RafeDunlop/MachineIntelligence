import numpy as np


def expected_cost(y_prob, bin_class, cost_mat):
    y_pos, y_neg = y_prob, 1 - y_prob
    return y_pos * cost_mat[1][bin_class] + y_neg * cost_mat[0][bin_class]


def min_cost_predictions(y_proba, cost_matrix):
    return np.fromiter(map(
        lambda y_prob: min([0, 1], key=lambda bin_class: expected_cost(y_prob, bin_class, cost_matrix)),
        y_proba
    ), dtype=int)

def test1():
    print()
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    cost_matrix = np.array([[0, 1],
                            [3, 0]])

    print(min_cost_predictions(y_proba, cost_matrix))

def test2():
    print()
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Load and split the data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=401)

    # Train a model
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]  # Probabilities for class 1

    cost_matrix = np.array([[0, 1],  # benign: TN = 0, FP = 1
                            [10, 0]])  # malignant: FN = 10, TP = 0

    preds = min_cost_predictions(y_proba, cost_matrix)

    # prediction distribution
    print(np.unique(preds, return_counts=True))

if __name__ == '__main__':
    test1()
    test2()