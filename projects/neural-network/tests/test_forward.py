import numpy as np
from nn.activations import ReLU
from nn.layers import Linear


def test_forward():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR input

    l1 = Linear(input_dim=2, output_dim=4, initializer="he")
    a1 = ReLU().forward(l1(X))

    l2 = Linear(input_dim=4, output_dim=2, initializer="he")
    y = ReLU().forward(l2(a1))

    print(y)


if __name__ == "__main__":
    test_forward()