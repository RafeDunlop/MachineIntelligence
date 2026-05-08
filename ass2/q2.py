from classes import LeafNode, SplitNode


def test1():
    leaf = LeafNode(True)
    print(leaf.leaf_count())

def test2():
    true_leaf = LeafNode(True)
    false_leaf = LeafNode(False)
    root = SplitNode(0)
    root.children = {0: true_leaf, 1: false_leaf}
    print(root.leaf_count())

if __name__ == "__main__":
    test2()