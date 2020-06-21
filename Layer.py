import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = []

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


if __name__ == "__main__":
    np.random.seed(0)

    X = [[1.0, 2.0, 3.0, 2.5],
         [2.0, 5.0, -1.0, 2.0],
         [-1.5, 2.7, 3.3, -0.8]]

    layer1 = Layer(4, 10)
    layer2 = Layer(10, 2)

    layer1.forward(X)
    layer2.forward(layer1.output)
    print(layer2.output)
