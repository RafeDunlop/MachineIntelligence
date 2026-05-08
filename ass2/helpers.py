from math import log2

def partition_by_feature_value(dataset, feature_index):
    partition = dict()
    for row in dataset:
        feature = row[0][feature_index]
        if feature not in partition:
            partition[feature] = []
        partition[feature].append(row)
    return partition

def class_proportions(dataset):
    classes = dict()
    for _, classification in dataset:
        if classification not in classes:
            classes[classification] = 0
        classes[classification] += 1
    return list(map(lambda count: count / len(dataset), classes.values()))

def misclassification(dataset):
    return 1 - max(class_proportions(dataset))

def gini(dataset):
    return 1 - sum([ratio ** 2 for ratio in class_proportions(dataset)])

def entropy(dataset):
    return -1 * sum([ratio * log2(ratio) for ratio in class_proportions(dataset)])
