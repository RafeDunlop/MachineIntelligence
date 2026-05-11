from free_classes import *
from q1 import ValueNode
from q2 import SumNode
from q3 import MatMulNode
from q4 import SigmoidNode


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

def test1():
    print()
    import numpy as np

    n = 1000  # number of data poitns
    d = 2  # number of features
    true_theta = np.array([3, 4])
    true_bias = 2

    # Synthetic binary classification data
    rng = np.random.default_rng(0)
    x_data = rng.uniform(low=-5, high=5, size=(n, d))
    y_data = ((true_bias + x_data @ true_theta) > 0).astype(int).reshape(-1, 1)

    # Create input and parameter nodes
    x = ValueNode(x_data)
    y = ValueNode(y_data)

    theta = ValueNode(np.zeros((d, 1)))
    bias = ValueNode(np.zeros((1, 1)))
    parameters = [theta, bias]

    # Build the rest of the graph for logistic regression
    tiled_bias = TileNode(bias, shape=(n, 1))
    logits = SumNode(MatMulNode(x, theta), tiled_bias)
    preds = SigmoidNode(logits)
    loss = LogLossNode(preds, y)

    train(loss, parameters, lr=0.1, epochs=50)

    # Compute final predictions and accuracy (on the same data)
    loss.propagate_forward()
    predictions = (preds.output > 0.5).astype(int)
    accuracy = np.mean(predictions == y_data)
    print(accuracy > 0.95)

if __name__ == '__main__':
    test1()