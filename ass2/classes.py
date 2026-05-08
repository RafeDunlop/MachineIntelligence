from abc import abstractmethod, ABC


class Node(ABC):
    @abstractmethod
    def predict(self, features):
        pass
    @abstractmethod
    def leaf_count(self):
        pass

class LeafNode(Node):
    def __init__(self, label):
        self.label = label
    def leaf_count(self):
        return 1
    def predict(self, features):
        return self.label

class SplitNode(Node):
    def __init__(self, feature_index):
        self.feature_index = feature_index
        self.children = dict()
    def leaf_count(self):
        return sum([child.leaf_count() for child in self.children.values()])
    def predict(self, features):
        return self.children[features[self.feature_index]].predict(features)