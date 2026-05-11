import numpy as np

from free_classes import TileNode, SoftmaxNode, CrossEntropyLossNode
from q1 import ValueNode
from q2 import SumNode
from q3 import MatMulNode
from q4 import SigmoidNode
from q5 import train


def build_network(x_data, y_data):

    np.random.seed(401)

    x_node = ValueNode(x_data)
    y_node = ValueNode(y_data)

    w1 = ValueNode(np.random.randn(x_data.shape[1], 32) * 0.1)
    b1 = ValueNode(np.zeros((1, 32)))

    w2 = ValueNode(np.random.randn(32, 10) * 0.1)
    b2 = ValueNode(np.zeros((1, 10)))

    parameters = [w1, b1, w2, b2]

    xw1 = MatMulNode(x_node, w1)

    z1 = SumNode(
        xw1,
        TileNode(b1, like=x_node)
    )

    a1 = SigmoidNode(z1)

    a1w2 = MatMulNode(a1, w2)

    z2 = SumNode(
        a1w2,
        TileNode(b2, like=x_node)
    )
    probs = SoftmaxNode(z2)
    loss = CrossEntropyLossNode(probs, y_node)

    return x_node, y_node, parameters, probs, loss

def test1():
    print()
    import numpy as np
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    def load_data():
        digits = load_digits()
        x = digits.data / 16.0  # Normalize pixel values
        y = digits.target.reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = encoder.fit_transform(y)
        return train_test_split(x, y_onehot, test_size=0.3, random_state=401)

    x_train, x_test, y_train, y_test = load_data()
    x_node, y_node, parameters, probs, loss = build_network(x_train, y_train)

    train(loss, parameters, lr=0.2, epochs=50)

    x_node.value = x_test
    probs.clear_computed()
    probs.propagate_forward()
    preds = np.argmax(probs.output, axis=1)

    truth = np.argmax(y_test, axis=1)
    acc = np.mean(preds == truth)
    print(acc > 0.2)

if __name__ == '__main__':
    test1()