# Dense Layer
import numpy as np
from pyparsing import dblSlashComment
from CreateData import create_data

class Layer_Dense:

    # Layer Initialization
    def __init__(self, n_inputs, n_neurons, weight_value=0.01):
        self.weights = weight_value * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backpropagation(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.dot(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)