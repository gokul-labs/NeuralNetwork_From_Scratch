from node import Node
import random


class Neuron:

    def __init__(self, no_of_inputs):
        self.w = [Node(random.uniform(-1, 1)) for _ in range(no_of_inputs)]
        self.b = Node(random.uniform(-1, 1))

    def __call__(self, x):
        s = sum([wi * xi for wi, xi in zip(self.w, x)], self.b)
        return s.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, in_nodes, out_nodes):
        self.neurons = [Neuron(in_nodes) for _ in range(out_nodes)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:

    def __init__(self, in_nodes, layer_specs):
        sizes = [in_nodes] + layer_specs
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_specs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
