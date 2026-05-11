from autodiff import Node
from q1 import ValueNode

class MatMulNode(Node):

    def get_in(self):
        A = self.inputs[0].output
        B = self.inputs[1].output
        return A, B

    def forward(self):
        A, B = self.get_in()
        assert A.shape[1] == B.shape[0]
        self.output = A @ B

    def backward(self):
        A, B = self.get_in()

        grad_A = self.grad @ B.T
        grad_B = A.T @ self.grad

        return [grad_A, grad_B]

def test1():
    print()
    import numpy as np

    A = ValueNode(np.array([[1.0, 2.0]]))  # shape (1, 2)
    B = ValueNode(np.array([[3.0], [4.0]]))  # shape (2, 1)

    mm = MatMulNode(A, B)

    mm.propagate_forward()
    print(mm.output)
    print()

    mm.grad = np.ones_like(mm.output)
    grads = mm.backward()
    print(grads[0])
    print()
    print(grads[1])

def test2():
    print()
    import numpy as np

    A = ValueNode(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    B = ValueNode(np.array([[1.0, 6.0, 0.0, -2.0], [-1.0, 2.0, 1.0, -3.0]]))

    mm = MatMulNode(A, B)

    mm.propagate_forward()
    print(mm.output)
    print()

    mm.grad = np.ones_like(mm.output)
    grads = mm.backward()
    print(grads[0])
    print()
    print(grads[1])

if __name__ == '__main__':
    test1()
    test2()
