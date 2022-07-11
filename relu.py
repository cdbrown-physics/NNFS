import numpy as np
class relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)