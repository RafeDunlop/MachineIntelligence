from classes import LeafNode, SplitNode

def main():
    t = LeafNode(True)
    f = LeafNode(False)
    n = SplitNode(0)
    n.children = {0: t, 1: f}

    print(n.predict((0,)))
    print(n.predict((1,)))

if __name__ == "__main__":
    main()

