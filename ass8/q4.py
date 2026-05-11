from autodiff import *
from q1 import ValueNode


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SigmoidNode(Node):

    def forward(self):
        self.output = sigmoid(self.inputs[0].output)

    def backward(self):
        grad_input = self.grad * self.output * (1 - self.output)
        return [grad_input]

def test1():
    print()
    import numpy as np

    np.set_printoptions(precision=4)

    x = ValueNode(np.array([[0.0, 2.0], [-1.0, 3.0]]))
    s = SigmoidNode(x)

    s.propagate_forward()
    print("Type of output:", type(s.output))
    print("Output:")
    print(s.output)
    print()

    s.grad = np.ones_like(s.output)  # Manually assign upstream gradient
    grads = s.backward()
    print("Type of the object returned by backward:", type(grads))
    print("Length of list:", len(grads))
    print("Type of the first and only element:", type(grads[0]))
    print("Shape:", grads[0].shape)
    print()
    print("The gradient wrt the input:")
    print(grads[0])

if __name__ == "__main__":
    test1()