from LayerDense import LayerDense
from CreateData import create_data
from ActivationReLU import ActivationReLU
import numpy as np


if __name__ == "__main__":
    np.random.seed(0)
    # Make data set
    dataPoints = 100
    arms = 3
    X, y = create_data(dataPoints, arms)

    dense1 = LayerDense(2, 3)

    activation1 = ActivationReLU()
    # Make a forward pass of the training data through this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    # Let's see output for first few samples:
    print(activation1.output[:5])
