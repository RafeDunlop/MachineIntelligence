from autodiff import Node
import numpy as np

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