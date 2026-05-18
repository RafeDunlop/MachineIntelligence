# Computational networks and automatic differentiation mini framework for COSC401
# Author: Kourosh Neshatian
# Last updated: 5 May 2025

import numpy as np

class Node:
    """Base class for all nodes in the computation graph.

    Each node represents a differentiable operation or
    value. Subclasses must implement their own `forward()` and
    `backward()` methods to define computation and gradient logic.
    Optionally, subclasses may also override `__init__()` to accept
    operation-specific inputs.

    """
    def __init__(self, *inputs):
        self.inputs = inputs
        assert(all(isinstance(input, Node) for input in self.inputs))
        self.output = None
        self.grad = None
        self._computed = False

    def propagate_forward(self):
        if self._computed:
            return
        for input_node in self.inputs:
            input_node.propagate_forward()
        self.forward()
        self._computed = True

    def propagate_backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.output)

        self.zero_grads()
        self.grad = grad

        topo_order = []
        
        def dfs(node, visited):
            if node not in visited:
                visited.add(node)
                for input_node in node.inputs:
                    dfs(input_node, visited)
                topo_order.append(node)

        dfs(self, set())

        for node in reversed(topo_order):
            grads_to_inputs = node.backward()
            for input_node, input_grad in zip(node.inputs, grads_to_inputs):
                input_node.grad += input_grad

    def forward(self):
        """Computes the output and stores it in self.output"""
        raise NotImplementedError("Each subclass must implement forward()")

    def backward(self):
        """Returns a list of gradients, one with respect to each input"""
        raise NotImplementedError("Each subclass must implement backward()")

    def clear_computed(self):
        def clear_fn(node):
            node._computed = False
        self.apply_to_graph(clear_fn)


    def zero_grads(self):
        def zero_fn(node):
            if not node._computed:
                # We need the output to know the shape of the gradient
                raise RuntimeError("Cannot zero gradients before forward pass")
                
            node.grad = np.zeros_like(node.output)
        self.apply_to_graph(zero_fn)


    def apply_to_graph(self, fn, visited=None):
        """Recursively applies a function to all nodes in the computation graph."""
        if visited is None:
            visited = set()
        if self in visited:
            return
        visited.add(self)
        fn(self)
        for input_node in self.inputs:
            input_node.apply_to_graph(fn, visited)


