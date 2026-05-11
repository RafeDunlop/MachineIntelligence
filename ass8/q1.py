from autodiff import Node
import numpy as np

class ValueNode(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self):
        self.output = self.value

    def backward(self):
        return np.array([])

def test1():
    print()
    import numpy as np

    value = np.array([[3.0, 4.0]])
    vnode = ValueNode(value)

    vnode.propagate_forward()
    print(np.allclose(vnode.output, value))

    print(len(vnode.backward()))

if __name__ == '__main__':
    test1()