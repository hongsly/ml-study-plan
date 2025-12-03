import numpy as np

def xavier_initialization(input_dim: int, output_dim: int) -> np.ndarray:
    # Uniform(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
    limit = np.sqrt(6/(input_dim + output_dim))
    return np.random.uniform(-limit, limit, size=(input_dim, output_dim))

def he_initialization(input_dim: int, output_dim: int) -> np.ndarray:
    # Normal(0, √(2/fan_in))
    std = np.sqrt(2/input_dim)
    return np.random.normal(0, std, size=(input_dim, output_dim))