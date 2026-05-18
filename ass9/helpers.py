import numpy as np
from autodiff import *

def train(loss_node, parameters, lr, epochs, print_loss=False):
    for epoch in range(epochs):

        # Clear previous forward computations
        loss_node.clear_computed()

        # Forward pass
        loss_node.propagate_forward()
        current_loss = loss_node.output

        # Zero gradients
        for param in parameters:
            param.grad = np.zeros_like(param.value)

        # Backward pass
        loss_node.grad = 1.0
        loss_node.propagate_backward()

        # Gradient descent update
        for param in parameters:
            param.value -= lr * param.grad

        if print_loss and epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {current_loss:.4f}")

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


# Paste all the class definitions here

class ValueNode(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self):
        self.output = self.value

    def backward(self):
        return np.array([])


class SumNode(Node):

    def forward(self):
        assert len(self.inputs) > 0
        shape = self.inputs[0].output.shape
        assert all(input.output.shape == shape for input in self.inputs)
        self.output = sum([input.output for input in self.inputs])

    def backward(self):
        return [self.grad.copy() for _ in self.inputs]


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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SigmoidNode(Node):

    def forward(self):
        self.output = sigmoid(self.inputs[0].output)

    def backward(self):
        grad_input = self.grad * self.output * (1 - self.output)
        return [grad_input]


class TileNode(Node):
    def __init__(self, node_to_tile, shape=None, like=None):
        super().__init__(node_to_tile)
        self.static_shape = shape
        self.like = like

    def forward(self):
        base = self.inputs[0].output
        if self.like is not None:
            n = self.like.output.shape[0]
            self.output = np.tile(base, (n, 1))
        else:
            self.output = np.tile(base, self.static_shape)

    def backward(self):
        # Tile repeats the input, so we need to sum over the repeated
        # dimensions. We assume tiling only along the first axis only.
        grad_input = np.sum(self.grad, axis=0, keepdims=True)
        return [grad_input]


class LogLossNode(Node):

    def forward(self):
        preds = self.inputs[0].output
        targets = self.inputs[1].output
        eps = 1e-9  # To prevent log(0)
        self.output = -np.mean(targets * np.log(preds + eps) + (1 - targets) * np.log(1 - preds + eps))

    def backward(self):
        preds = self.inputs[0].output
        targets = self.inputs[1].output
        n = targets.shape[0]
        eps = 1e-9
        grad_pred = (-targets / (preds + eps) + (1 - targets) / (1 - preds + eps)) / n
        return [self.grad * grad_pred, np.zeros_like(targets)]


class SoftmaxNode(Node):

    def forward(self):
        logits = self.inputs[0].output
        # Shift for numerical stability; Result is invariant to shift!
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        self.output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def backward(self):
        probs = self.output
        grad = self.grad
        dot = np.sum(probs * grad, axis=1, keepdims=True)
        grad_input = probs * (grad - dot)
        return [grad_input]


class CrossEntropyLossNode(Node):

    def forward(self):
        probs = self.inputs[0].output
        targets = self.inputs[1].output
        eps = 1e-9
        self.output = -np.mean(np.sum(targets * np.log(probs + eps), axis=1))

    def backward(self):
        probs = self.inputs[0].output
        targets = self.inputs[1].output
        batch_size = probs.shape[0]
        grad_probs = -targets / (probs + 1e-9) / batch_size
        return [grad_probs * self.grad, np.zeros_like(targets)]


