import numpy as np

class softmax:
    def forward(self, inputs):
        #un-normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        prob = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = prob