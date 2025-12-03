import numpy as np
from nn.module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (batch_size, dim)
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad: (batch_size, dim)
        return grad * (self.input > 0)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return 1 / (1 + np.exp(-x))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (np.exp(-self.input) / (1 + np.exp(-self.input)) ** 2)


class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # numerically stable softmax by subtracting the max value
        max_value = np.max(x, axis=1, keepdims=True)
        exp_values = np.exp(x - max_value)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # TODO
        pass