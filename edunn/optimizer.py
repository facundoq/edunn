# Additional material to help you implement optimizers:
# "An overview of gradient descent optimization algorithms" https://ruder.io/optimizing-gradient-descent/


from typing import Dict, Optional
import numpy as np
from .model import Model, Phase
from .model import ParameterSet
import sys, abc
from tqdm.auto import tqdm


class Optimizer(abc.ABC):

    @abc.abstractmethod
    def optimize_batch(self, model: Model, δEδps: ParameterSet, epoch: int, iteration: int):
        pass


def all_equal(list: []):
    return len(list) == 0 or list.count(list[0]) == len(list)


import random


def batch_arrays(batch_size: int, *arrays, shuffle=False):
    """

    :param batch_size: size of batches
    :param arrays: variable number of numpy arrays
    :return: a generator that returns the arrays in batches
    """

    sample_sizes = [a.shape[0] for a in arrays]
    assert all_equal(sample_sizes)
    batches = sample_sizes[0] // batch_size
    batch_list = list(range(batches))
    if shuffle:
        random.shuffle(batch_list)
    for i in batch_list:
        start = i * batch_size
        end = start + batch_size
        batch = [a[start:end,] for a in arrays]
        yield tuple(batch)


class GradientDescent(Optimizer):

    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def optimize_batch(self, model: Model, δEδps: ParameterSet, epoch: int, iteration: int):

        # Update parameters
        parameters = model.get_parameters()
        for parameter_name, δEδp in δEδps.items():
            p = parameters[parameter_name]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            """YOUR IMPLEMENTATION START"""
            p[:] = p - self.lr * δEδp
            """YOUR IMPLEMENTATION END"""


class RMSprop(Optimizer):

    def __init__(
        self, lr: float = 0.1, beta: float = 0.99, eps: float = 1e-8
    ):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.first = True
        self.v = {}

    def optimize_batch(self, model: Model, δEδps: ParameterSet, epoch: int, iteration: int):
        if self.first:
            self.first = False
            for k, p in model.get_parameters().items():
                self.v[k] = np.zeros_like(p)

        # Update parameters
        parameters = model.get_parameters()
        for parameter_name, δEδp in δEδps.items():
            p = parameters[parameter_name]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            """YOUR IMPLEMENTATION START"""
            self.v[parameter_name] = self.beta * self.v[parameter_name] + (1 - self.beta) * δEδp * δEδp
            p[:] = p - self.lr / (np.sqrt(self.v[parameter_name]) + self.eps) * δEδp
            """YOUR IMPLEMENTATION END"""


class Adam(Optimizer):

    def __init__(
        self, lr: float = 0.1, betas: tuple = (0.9, 0.999), eps: float = 1e-08
    ):
        self.lr = lr
        self.beta_1, self.beta_2 = betas
        self.eps = eps
        self.first = True
        self.m = {}
        self.v = {}

    def optimize_batch(self, model: Model, δEδps: ParameterSet, epoch: int, iteration: int):
        if self.first:
            self.first = False
            for k, p in model.get_parameters().items():
                self.m[k] = np.zeros_like(p)
                self.v[k] = np.zeros_like(p)
        iteration += 1

        # Update parameters
        parameters = model.get_parameters()
        for parameter_name, δEδp in δEδps.items():
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


class MomentumGD(Optimizer):

    def __init__(self, lr: float = 0.1, gamma=0.9):
        self.lr = lr
        self.gamma = gamma
        self.first = True
        self.v = {}

    def optimize_batch(self, model: Model, δEδps: ParameterSet, epoch: int, iteration: int):
        if self.first:
            self.first = False
            for k, p in model.get_parameters().items():
                self.v[k] = np.zeros_like(p)

        # Update parameters
        parameters = model.get_parameters()
        for k, δEδp in δEδps.items():
            # K = parameter name
            p = parameters[k]
            v = self.v[k]
            # use p[:] and v[:] so that updates are in-place
            # instead of creating a new variable
            """YOUR IMPLEMENTATION START"""
            v[:] = self.gamma * v + self.lr * δEδp
            p[:] = p - v
            """YOUR IMPLEMENTATION END"""


class NesterovMomentumGD(Optimizer):

    def __init__(self, lr: float = 0.1, gamma=0.9):
        self.lr = lr
        self.gamma = gamma
        self.first = True
        self.v = {}

    def optimize_batch(self, model: Model, δEδps: ParameterSet, epoch: int, iteration: int):
        if self.first:
            self.first = False
            for k, p in model.get_parameters().items():
                self.v[k] = np.zeros_like(p)

        # Update parameters
        parameters = model.get_parameters()
        for k, δEδp in δEδps.items():
            # K = parameter name
            p = parameters[k]
            v = self.v[k]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            """YOUR IMPLEMENTATION START"""
            v[:] = self.gamma * v + self.lr * δEδp
            p[:] = p - (self.gamma * v + self.lr * δEδp)
            """YOUR IMPLEMENTATION END"""


class SignGD(Optimizer):

    def __init__(self, lr: float = 0.1, eps: float = 1e-8):
        self.eps = eps
        self.lr = lr

    def optimize_batch(self, model: Model, δEδps: ParameterSet, epoch: int, iteration: int):

        # Update parameters
        parameters = model.get_parameters()
        for parameter_name, δEδp in δEδps.items():
            p = parameters[parameter_name]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            """YOUR IMPLEMENTATION START"""
            denom = np.sqrt(δEδp**2 + self.eps)
            p[:] = p - self.lr * (δEδp / denom)
            """YOUR IMPLEMENTATION END"""
