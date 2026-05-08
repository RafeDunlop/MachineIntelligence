from classes import SplitNode, LeafNode
from helpers import partition_by_feature_value, misclassification


def classify(dataset):
    classes = dict()
    for _, classification in dataset:
        if classification not in classes:
            classes[classification] = 0
        classes[classification] += 1
    return max(classes, key=classes.get)

def delta_impurity(partition, dataset, imp):
    imp_split = sum(imp(q) * len(q) for q in partition.values()) / len(dataset)
    return imp(dataset) - imp_split


def train_tree(dataset, impurity_measure, features = None):
    sample_feature, default_label = dataset[0]
    if features is None:
        features = set(range(len(sample_feature)))
    if len(features) == 0:
        print(classify(dataset))
        return LeafNode(classify(dataset))
    if all(label == default_label for _, label in dataset):
        print(classify(dataset))
        return LeafNode(default_label)
    feature_partitions = [(i, partition_by_feature_value(dataset, i)) for i in features]
    key_fn = lambda part: delta_impurity(part[1], dataset, impurity_measure)
    feature_index, part_star = max(feature_partitions, key=key_fn)
    if delta_impurity(part_star, dataset, impurity_measure) < 0:
        print(classify(dataset))
        return LeafNode(classify(dataset))
    root = SplitNode(feature_index)
    root.children = {feature: train_tree(subset, impurity_measure, features - {feature_index}) for feature, subset in part_star.items()}
    return root

def test1():
    dataset = [
        ((True, True), False),
        ((True, False), True),
        ((False, True), True),
        ((False, False), False)
    ]
    t = train_tree(dataset, misclassification)
    print()
    print(t.predict((True, False)))
    print(t.predict((False, False)))

if __name__ == '__main__':
    test1()