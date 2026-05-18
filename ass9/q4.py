import numpy as np
from helpers import *
from autodiff import Node
from q3 import autoencoder_widths

class MSELossNode(Node):
    def forward(self):
        preds = self.inputs[0].output
        targets = self.inputs[1].output
        self.output = np.mean((preds - targets) ** 2)

    def backward(self):
        preds = self.inputs[0].output
        targets = self.inputs[1].output
        n = preds.shape[0]
        grad_preds = 2 * (preds - targets) / n
        return [grad_preds * self.grad, np.zeros_like(targets)]


class ReLUNode(Node):
    def forward(self):
        x = self.inputs[0].output
        self.output = np.maximum(0, x)

    def backward(self):
        x = self.inputs[0].output
        mask = x > 0
        return [self.grad * mask]

def build_autoencoder(input_data, k, ratio):
    np.random.seed(0)

    d = input_data.shape[1]
    widths = autoencoder_widths(d, k, ratio)

    input = ValueNode(input_data)
    target = input                                                                                                             
    parameters = []
    activations = []
    a = input

    # iterating over 2-grams
    for p, q in zip(widths, widths[1:]):
        w = ValueNode(np.random.randn(p, q) * 0.1) # Replace 0, 0
        b = ValueNode(np.zeros((1, q))) # Replace 0
        parameters.extend([w, b])
        z = SumNode(MatMulNode(a, w), TileNode(b, like=a))
        a = ReLUNode(z)
        activations.append(a)

    # Linear layer for regresssion (using the last z)                                                                                                          
    output = z
    loss = MSELossNode(output, target)

    return input, output, activations, parameters, loss


def test1():
    print()
    import numpy as np

    data = np.ones((2, 64))
    input, output, activations, parameters, loss = build_autoencoder(data, 4, 2.5)

    output.clear_computed()
    output.propagate_forward()

    print(type(input).__name__)
    print(input.output.shape)
    print()

    for node in activations[:-1]:
        print(type(node).__name__)
        print(node.output.shape)
        print()

    print(type(output).__name__)
    print(output.output.shape)

    np.set_printoptions(precision=5)
    print(output.output[:, :5])

if __name__ == '__main__':
    test1()
