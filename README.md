# Kolmogorov Arnold Neural Networks Framework

## Introduction

This is a Python implementation of the Kolmogorov-Arnold Neural Networks Framework. The framework is based on the Kolmogorov-Arnold Representation Theorem, which states that any continuous function can be represented as a composition of a finite number of elementary functions. The framework provides a set of elementary functions that can be combined to create complex neural networks for regression and classification tasks.

## How to use

Using the framework is fairly simple. First, you need to import the necessary modules:

```python
from src.kan import KAN_Layer
```

Next, to initialize a KAN layer, you need to specify the number of input and output neurons, the knots dimension, and the B-spline generator:

```python
kan = KAN_Layer(in_features=2, out_features=3, knots_dim=8, bsplines_generator=lambda x, k: torch.functional.F.relu(x - k))
```

You can then use the KAN layer as a regular PyTorch layer:

```python
x = torch.randn(10, 2)
y = kan(x)
```

## Examples

The `log_example.py` file contains an example of how to use the KAN framework for approximating the log function. The example demonstrates how to create a simple neural network using KAN layers and train it on a synthetic dataset.

## References

- Kolmogorov, A. N. (1957). On the representation of continuous functions of many variables by superpositions of continuous functions of one variable and addition. Doklady Akademii Nauk SSSR, 114(5), 953-956.
- Arnold, V. I. (1957). On functions of three variables. Doklady Akademii Nauk SSSR, 114(4), 679-681.
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 2(4), 303-314.
- Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. Neural networks, 2(5), 359-366.
- Funahashi, K. I. (1989). On the approximate realization of continuous mappings by neural networks. Neural networks, 2(3), 183-192.
- Leshno, M., Lin, V. Y., Pinkus, A., & Schocken, S. (1993). Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. Neural networks, 6(6), 861-867.
- Barron, A. R. (1993). Universal approximation bounds for superpositions of a sigmoidal function. IEEE Transactions on Information Theory, 39(3), 930-945.
- Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. Neural networks, 4(2), 251-257.
- iu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2025). KAN: Kolmogorov-Arnold Networks. arXiv. <https://arxiv.org/abs/2404.19756>
