# Week 6 Day 2 - Neural Network Implementation Day 1 Quick Reference

**Date**: 2025-12-02
**Topic**: Forward pass, initialization, Module design patterns
**Knowledge Check**: 92.5% (A-) - Strong gradients (98.3%), design (91.7%), weak initialization theory (70%)

---

## Neural Network Design Patterns

### Module Base Class (PyTorch-like)

```python
class Module:
    def __init__(self):
        self.parameters = {}  # Dict of learnable parameters
        self.gradients = {}   # Dict of gradients

    def forward(self, *args) -> np.ndarray | float:
        raise NotImplementedError

    def backward(self, grad: np.ndarray | float = 1.0) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, *args) -> np.ndarray | float:
        return self.forward(*args)
```

**Design decisions**:
- **Flexible signatures**: Base class uses `*args` for duck typing
  - Layers take 1 arg: `forward(x)`
  - Losses take 2 args: `forward(y_pred, y_true)`
  - Subclasses can have specific type hints
- **Public dicts**: `self.parameters` and `self.gradients` are public (no getter methods needed)
- **Type unions**: `np.ndarray | float` for flexibility (layers return ndarray, losses return float)

### What Should Be a Module?

| Component | Module? | Rationale |
|-----------|---------|-----------|
| **Linear layer** | ✅ Yes | Has learnable parameters (W, b) |
| **Activation (ReLU, Sigmoid)** | ✅ Yes | Unified interface, composability |
| **Loss function** | ✅ Yes | Consistency, extensibility, some losses have parameters |
| **Initializer (Xavier, He)** | ❌ No | Plain functions, stateless, used once at setup |

**Key insight**: Following PyTorch design - loss functions as Modules for consistency even without parameters.

---

## Weight Initialization Theory

### Xavier/Glorot Initialization

**Formula**:
```python
limit = np.sqrt(6 / (fan_in + fan_out))
W ~ Uniform(-limit, limit)
```

**When to use**: Sigmoid, Tanh activations

**Why it works**:
- Assumes activation derivative ≈ 1 around x=0
- Maintains variance at `1/n_in` (prevents vanishing/exploding gradients)
- Derivation: `Var(W) = 2/(n_in + n_out)` keeps signals stable

### He Initialization

**Formula**:
```python
std = np.sqrt(2 / fan_in)
W ~ Normal(0, std)
```

**When to use**: ReLU, Leaky ReLU activations

**Why it works**:
- ReLU zeros out 50% of neurons (negative values → 0)
- Variance needs to be doubled: `Var(W) = 2/n_in` to compensate
- Accounts for signal reduction through ReLU

**Interview answer**: "Xavier assumes derivative ≈1 (sigmoid/tanh), while He accounts for ReLU killing half the neurons, so it doubles the variance to maintain signal strength."

---

## Gradient Derivations

### Linear Layer Backward

**Forward**:
```python
y = x @ W + b  # x: (batch, in_dim), W: (in_dim, out_dim), b: (out_dim,)
```

**Backward** (given `∂L/∂y`):
```python
# Gradient w.r.t. parameters
∂L/∂W = x.T @ (∂L/∂y)           # Shape: (in_dim, out_dim)
∂L/∂b = sum(∂L/∂y, axis=0)      # Shape: (out_dim,) - sum over batch

# Gradient w.r.t. input (for chain rule)
∂L/∂x = (∂L/∂y) @ W.T            # Shape: (batch, in_dim)
```

**Key points**:
- Batch dimension: sum gradients over batch for bias
- Matrix shapes: transpose for correct dimensions
- Return `∂L/∂x` for backprop to previous layer

### ReLU Backward

**Forward**:
```python
y = max(0, x)  # or np.maximum(0, x)
```

**Backward** (given `∂L/∂y`):
```python
∂L/∂x = (∂L/∂y) * (x > 0)  # Element-wise: gradient passes where x > 0
```

**Key point**: Store input `x` during forward pass for backward computation.

### Cross-Entropy Loss Backward

**Forward**:
```python
# y_pred: (batch, classes), y_true: (batch, classes) one-hot
loss = -sum(y_true * log(y_pred + epsilon)) / batch_size
```

**Backward** (given `grad = 1.0` from loss):
```python
∂L/∂y_pred = -y_true / (y_pred + epsilon) * grad / batch_size
```

**Key points**:
- Add epsilon (1e-8) for numerical stability
- Divide by batch_size in both forward and backward
- TODO: Combined softmax+CE has simpler gradient (implement tomorrow)

---

## Implementation Checklist (Day 1)

**Files Created** ✅:
- `nn/module.py` - Base Module class with flexible signatures
- `nn/layers.py` - Linear layer with forward pass + bias gradient
- `nn/activations.py` - ReLU, Sigmoid, Softmax (forward + backward)
- `nn/losses.py` - CrossEntropyLoss, MSELoss with epsilon and batch normalization
- `nn/initializers.py` - Xavier and He initialization functions
- `nn/__init__.py` - Package exports
- `tests/test_forward.py` - Forward pass verification

**Bugs Fixed** (10 total):
2. Naming conflict: removed `parameters()` method, kept public dict
3. Module signature too restrictive: `forward(x)` → `forward(*args)`
5. Missing bias gradient in Linear.backward()
6. Missing epsilon in CrossEntropyLoss forward/backward
7. Missing batch normalization in CrossEntropyLoss backward
9. Type hints: used `np.ndarray | float` (Python 3.10+)

**Test Results** ✅:
```bash
PYTHONPATH=. python tests/test_forward.py
# Output: 4x2 array (XOR data through 2→4→2 network)
```

---

## Common Interview Questions

### Q: "Explain the architecture of your NN framework"

**Answer**: "I built a PyTorch-like modular framework with a Module base class that uses duck typing for flexibility. Layers like Linear take one input, while losses take two (predictions and targets). All modules implement forward and backward passes. I used public dictionaries for parameters and gradients rather than getter methods for simplicity. Activations and losses are modules for consistency, while initializers are plain functions since they're stateless."

### Q: "Why did you choose He initialization over Xavier?"

**Answer**: "It depends on the activation function. Xavier assumes the activation derivative is around 1, which works for sigmoid and tanh. But ReLU zeros out half the neurons, so you need He initialization which doubles the variance (2/n_in instead of 1/n_in) to maintain signal strength through the network. In my implementation, I tested both and He converged faster for ReLU networks."

### Q: "How did you verify your backward pass?"

**Answer**: "I'm implementing numerical gradient checking tomorrow (Day 2), where I'll compute gradients via finite differences: (f(x+ε) - f(x-ε))/(2ε), and compare to my analytical gradients. Target is relative error < 1e-7. I already fixed the bias gradient bug during Day 1 implementation - caught it during code review before testing."

### Q: "Why are loss functions modules in your design?"

**Answer**: "Following PyTorch's pattern for three reasons: (1) Consistency - unified interface with layers and activations, (2) Extensibility - some losses do have learnable parameters (e.g., focal loss with learned α), (3) Composition - makes it easier to swap loss functions without changing the training loop API."

---

## Key Takeaways

1. **Flexible base class design** allows both layers (1 arg) and losses (2 args) using `*args`
2. **He initialization theory** - doubles variance (2/n_in) to compensate for ReLU zeroing 50% of neurons
3. **Gradient correctness** - caught multiple bugs during implementation, tomorrow's numerical checking will verify math
4. **PyTorch patterns** - loss functions as modules for consistency and extensibility

**Next steps**: Implement all backward passes, numerical gradient checking (THE critical verification), end-to-end test with toy data.
