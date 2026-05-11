from autodiff import Node
from q1 import ValueNode
import numpy as np

class SumNode(Node):

    def forward(self):
        assert len(self.inputs) > 0
        shape = self.inputs[0].output.shape
        assert all(input.output.shape == shape for input in self.inputs)
        self.output = sum([input.output for input in self.inputs])

    def backward(self):
        return [self.grad.copy() for _ in self.inputs]

def test1():
    print()
    import numpy as np

    x = ValueNode(np.array([[1.0, 2.0]]))
    y = ValueNode(np.array([[3.0, 4.0]]))
    z = ValueNode(np.array([[5.0, 6.0]]))

    s = SumNode(x, y, z)

    s.propagate_forward()
    print(s.output)

    s.grad = np.ones_like(s.output)
    grads = s.backward()
    print(len(grads))
    print(all(np.allclose(g, np.ones_like(s.output)) for g in grads))

if __name__ == '__main__':
    test1()