# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 14:30:09 2022

@author: NinjaOfPhysics
"""

import numpy as np
import nnfs
from nnfs.datasets import vertical_data 

import Layer
import Loss
from relu import relu
from softmax import softmax

nnfs.init()

if __name__ == "__main__":
    X, y = vertical_data(samples=100, classes=3)
    dense1 = Layer.Layer_Dense(2,3)
    dense2 = Layer.Layer_Dense(3,3)

    activation1 = relu()
    activation2 = softmax()

    loss_function = Loss.Loss_CategoricalCrossentropy()

    lowest_loss = 999999
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.weights.copy()

    for iteration in range(10000):
        dense1.weights += 0.05 * np.random.randn(2,3)
        dense1.biases += 0.05 * np.random.randn(1,3)
        dense2.weights += 0.05 * np.random.randn(3,3)
        dense2.biases += 0.05 * np.random.randn(1,3)

        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        loss = loss_function.calculate(activation2.output, y)
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)

        if loss < lowest_loss:
            print("New set of weights found, iteration:", iteration,
                  "loss: ", loss,
                  "acc:", accuracy)
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss
        else:
            dense1.weights = best_dense1_weights.copy()
            dense1.biases = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases = best_dense2_biases.copy()
