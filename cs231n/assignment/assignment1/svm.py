import numpy as np


def linear_func(W, x):
    """linear function"""
    return np.dot(W, x)


class SVM:

    def __init__(self, step=10000, learning_rate=0.01):
        self.step = step
        self.learning_rate = 0.01

    def train(self, X, y):
