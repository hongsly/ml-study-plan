import numpy as np
from nn.module import Module
from nn.initializers import xavier_initialization, he_initialization


class Linear(Module):
    """A linear layer."""

    def __init__(self, input_dim: int, output_dim: int, initializer: str = "xavier"):
        super().__init__()

        match initializer:
            case "xavier":
                self.parameters["W"] = xavier_initialization(input_dim, output_dim)
            case "he":
                self.parameters["W"] = he_initialization(input_dim, output_dim)
            case _:
                self.parameters["W"] = np.random.randn(input_dim, output_dim)

        self.parameters["b"] = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (batch_size, input_dim)
        self.input = x
        return x @ self.parameters["W"] + self.parameters["b"]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad: (batch_size, output_dim)
        self.gradients["W"] = self.input.T @ grad
        self.gradients["b"] = np.sum(grad, axis=0)
        return grad @ self.parameters["W"].T
