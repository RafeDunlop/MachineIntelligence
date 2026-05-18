import numpy as np
from autodiff import Node
from q4 import *
from q3 import autoencoder_widths

def sim_rank(x, xs, input_node, activation_nodes):
    latent_node = activation_nodes[len(activation_nodes) // 2]
    def encode(sample):
        input_node.value = sample
        latent_node.clear_computed()
        latent_node.propagate_forward()
        return latent_node.output
    z_x, z_xs = encode(x), encode(xs)
    dists = np.sum((z_xs - z_x) ** 2, axis=1)
    return sorted(
        range(len(xs)),
        key=lambda i: dists[i]
    )

def test1():
    print()
    from sklearn.datasets import load_digits

    digits = load_digits()
    x_data, class_data = digits.data / 16.0, digits.target
    # Class data will not be used!

    input, ouput, activations, parameters, loss = build_autoencoder(x_data, 4, 2.5)

    train(loss, parameters, lr=0.05, epochs=1000)

    print(sim_rank(x_data[0:1], x_data[0:20], input, activations))
    print(sim_rank(x_data[1:2], x_data[0:20], input, activations))
    print(sim_rank(x_data[5:6], x_data[50:60], input, activations))

if __name__ == '__main__':
    test1()
