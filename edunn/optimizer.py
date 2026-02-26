# Additional material to help you implement optimizers:
# "An overview of gradient descent optimization algorithms" https://ruder.io/optimizing-gradient-descent/


import numpy as np
from .model import Model, ParameterSet
import abc


class Optimizer(abc.ABC):
    """
    Base class for optimization algorithms.

    An Optimizer defines *how* to update model parameters given their gradients.
    It does NOT handle the training loop (batching, epochs, etc.) -- that is the
    responsibility of a Trainer (see trainers.py).

    Subclasses must implement:
        - step(parameters, gradients, epoch, iteration): apply a single parameter update
    Subclasses may optionally override:
        - initialize(parameters): set up internal state (momentum buffers, etc.)
    """

    def initialize(self, parameters: ParameterSet):
        """
        Initialize any internal state needed by the optimizer (e.g., momentum buffers).
        Called once before the first call to step().

        :param parameters: dictionary mapping parameter names to numpy arrays
        """
        pass

    @abc.abstractmethod
    def step(self, parameters: ParameterSet, gradients: ParameterSet, epoch: int, iteration: int):
        """
        Perform a single parameter update.

        :param parameters: dictionary mapping parameter names to numpy arrays (mutable, update in-place)
        :param gradients: dictionary mapping parameter names to their gradient arrays
        :param epoch: current epoch number
        :param iteration: current iteration (batch) number within the epoch
        """
        pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Updates parameters using the rule:
        p = p - lr * gradient
    """

    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def step(self, parameters: ParameterSet, gradients: ParameterSet, epoch: int, iteration: int):
        for parameter_name, δEδp in gradients.items():
            p = parameters[parameter_name]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            """YOUR IMPLEMENTATION START"""
            p[:] = p - self.lr * δEδp
            """YOUR IMPLEMENTATION END"""


class MomentumSGD(Optimizer):
    """
    Gradient Descent with Momentum.

    Maintains a velocity buffer and updates parameters using:
        v = gamma * v + lr * gradient
        p = p - v
    """

    def __init__(self, lr: float = 0.1, gamma: float = 0.9):
        self.lr = lr
        self.gamma = gamma
        self.v = {}

    def initialize(self, parameters: ParameterSet):
        for k, p in parameters.items():
            self.v[k] = np.zeros_like(p)

    def step(self, parameters: ParameterSet, gradients: ParameterSet, epoch: int, iteration: int):
        if not self.v:
            self.initialize(parameters)

        for k, δEδp in gradients.items():
            p = parameters[k]
            v = self.v[k]
            # use p[:] and v[:] so that updates are in-place
            # instead of creating a new variable
            """YOUR IMPLEMENTATION START"""
            v[:] = self.gamma * v + self.lr * δEδp
            p[:] = p - v
            """YOUR IMPLEMENTATION END"""


class NesterovMomentumSGD(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG) optimizer.

    A variant of momentum that "looks ahead" by computing the gradient at
    the anticipated future position:
        v = gamma * v + lr * gradient
        p = p - (gamma * v + lr * gradient)
    """

    def __init__(self, lr: float = 0.1, gamma: float = 0.9):
        self.lr = lr
        self.gamma = gamma
        self.v = {}

    def initialize(self, parameters: ParameterSet):
        for k, p in parameters.items():
            self.v[k] = np.zeros_like(p)

    def step(self, parameters: ParameterSet, gradients: ParameterSet, epoch: int, iteration: int):
        if not self.v:
            self.initialize(parameters)

        for k, δEδp in gradients.items():
            p = parameters[k]
            v = self.v[k]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            """YOUR IMPLEMENTATION START"""
            v[:] = self.gamma * v + self.lr * δEδp
            p[:] = p - (self.gamma * v + self.lr * δEδp)
            """YOUR IMPLEMENTATION END"""


class RMSpropOptimizer(Optimizer):
    """
    RMSprop optimizer.

    Adapts the learning rate per-parameter using a running average of
    squared gradients:
        v = beta * v + (1 - beta) * gradient^2
        p = p - lr / (sqrt(v) + eps) * gradient
    """

    def __init__(self, lr: float = 0.1, beta: float = 0.99, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = {}

    def initialize(self, parameters: ParameterSet):
        for k, p in parameters.items():
            self.v[k] = np.zeros_like(p)

    def step(self, parameters: ParameterSet, gradients: ParameterSet, epoch: int, iteration: int):
        if not self.v:
            self.initialize(parameters)

        for parameter_name, δEδp in gradients.items():
            p = parameters[parameter_name]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            """YOUR IMPLEMENTATION START"""
            self.v[parameter_name] = self.beta * self.v[parameter_name] + (1 - self.beta) * δEδp * δEδp
            p[:] = p - self.lr / (np.sqrt(self.v[parameter_name]) + self.eps) * δEδp
            """YOUR IMPLEMENTATION END"""


class AdamOptimizer(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).

    Combines ideas from momentum and RMSprop, maintaining both first and
    second moment estimates of the gradients with bias correction:
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        p = p - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, lr: float = 0.1, betas: tuple = (0.9, 0.999), eps: float = 1e-08):
        self.lr = lr
        self.beta_1, self.beta_2 = betas
        self.eps = eps
        self.m = {}
        self.v = {}

    def initialize(self, parameters: ParameterSet):
        for k, p in parameters.items():
            self.m[k] = np.zeros_like(p)
            self.v[k] = np.zeros_like(p)

    def step(self, parameters: ParameterSet, gradients: ParameterSet, epoch: int, iteration: int):
        if not self.m:
            self.initialize(parameters)
        iteration += 1

        for parameter_name, δEδp in gradients.items():
            p = parameters[parameter_name]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            """YOUR IMPLEMENTATION START"""
            self.m[parameter_name] = self.beta_1 * self.m[parameter_name] + (1 - self.beta_1) * δEδp
            self.v[parameter_name] = self.beta_2 * self.v[parameter_name] + (1 - self.beta_2) * δEδp * δEδp
            m_hat = self.m[parameter_name] / (1 - np.power(self.beta_1, iteration))
            v_hat = self.v[parameter_name] / (1 - np.power(self.beta_2, iteration))
            p[:] = p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            """YOUR IMPLEMENTATION END"""


class SignSGDOptimizer(Optimizer):
    """
    Sign Stochastic Gradient Descent.

    Normalizes each gradient component by its magnitude, effectively
    taking a unit step in the direction of the gradient:
        p = p - lr * gradient / sqrt(gradient^2 + eps)
    """

    def __init__(self, lr: float = 0.1, eps: float = 1e-8):
        self.lr = lr
        self.eps = eps

    def step(self, parameters: ParameterSet, gradients: ParameterSet, epoch: int, iteration: int):
        for parameter_name, δEδp in gradients.items():
            p = parameters[parameter_name]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            """YOUR IMPLEMENTATION START"""
            denom = np.sqrt(δEδp**2 + self.eps)
            p[:] = p - self.lr * (δEδp / denom)
            """YOUR IMPLEMENTATION END"""
