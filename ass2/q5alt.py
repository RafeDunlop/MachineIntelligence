from ass2.classes import LeafNode, SplitNode
from helpers import partition_by_feature_value, misclassification


def split_imp(subsets, impurity_measure):
    total_card = sum(len(subset) for subset in subsets)
    return sum(len(subset) * impurity_measure(subset) for subset in subsets) / total_card

def majority_label(dataset):
    counts = dict()
    for _, label in dataset:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return max(counts, key=counts.get)

def train_tree(dataset, impurity_measure, features = None):
    first_label = dataset[0][1]
    all_features = range(len(dataset[0][0]))
    if features is None:
        features = set(all_features)
    if all(label == first_label for _, label in dataset):
        return LeafNode(first_label)
    if not features:
        return LeafNode(majority_label(dataset))
    imp = impurity_measure(dataset)
    partitions = {feature: partition_by_feature_value(dataset, feature) for feature in features}
    delta_imps = {feature: imp - split_imp(partitions[feature].values(), impurity_measure) for feature in features}
    feat_star = max(features, key=delta_imps.get)
    node = SplitNode(feat_star)
    node.children = {
        feature_value: train_tree(subset, impurity_measure, features - {feat_star})
        for feature_value, subset in partitions[feat_star].items()
    }
    return node

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




