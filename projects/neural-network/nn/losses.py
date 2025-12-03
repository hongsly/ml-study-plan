import numpy as np
from nn.module import Module


class CrossEntropyLoss(Module):
    # Cros-entropy loss on softmax output
    # TODO: implement combined SoftmaxCrossEntropyLoss on logits output

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # y_pred: (batch_size, num_classes)
        # y_true: (batch_size, num_classes)
        epsilon = 1e-8
        self.y_pred = y_pred
        self.y_true = y_true
        batch_size = y_pred.shape[0]
        return -np.sum(y_true * np.log(y_pred + epsilon)) / batch_size

    def backward(self, grad: float = 1.0) -> np.ndarray:
        # return shape: (batch_size, num_classes)
        epsilon = 1e-8
        batch_size = self.y_pred.shape[0]
        return -self.y_true / (self.y_pred + epsilon) * grad / batch_size


class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # y_pred: (batch_size, 1)
        # y_true: (batch_size, 1)
        batch_size = y_pred.shape[0]
        self.y_pred = y_pred
        self.y_true = y_true
        return np.sum((y_pred - y_true) ** 2) / batch_size

    def backward(self, grad: float = 1.0) -> np.ndarray:
        batch_size = self.y_pred.shape[0]
        # return shape: (batch_size, 1)
        return 2 * (self.y_pred - self.y_true) / batch_size * grad
