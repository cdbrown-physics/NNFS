# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 14:30:09 2022

@author: NinjaOfPhysics
"""

import numpy as np
import nnfs
from nnfs.datasets import spiral_data 

import Layer
from relu import relu
from softmax import softmax

nnfs.init()

if __name__ == "__main__":
    X, y = spiral_data(samples=100, classes=3)
    dense1 = Layer.Layer_Dense(2,3)
    dense2 = Layer.Layer_Dense(3,3)

    activation1 = relu()
    activation2 = softmax()

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    print(activation2.output[:5])
