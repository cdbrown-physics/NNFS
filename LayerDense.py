import numpy as np
from CreateData import create_data


class LayerDense:

    def __init__(self, inputs, neurons, weight_value=0.01):
        self.weights = weight_value * np.random.randn(inputs, neurons)
        self.biases = np.zeros(shape=(1, neurons))
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs



