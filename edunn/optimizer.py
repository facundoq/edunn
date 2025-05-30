# Additional material to help you implement optimizers:
# "An overview of gradient descent optimization algorithms" https://ruder.io/optimizing-gradient-descent/


import abc
import random
import sys

import numpy as np
from tqdm.auto import tqdm

from .model import Model, Phase
from .model import ParameterSet


class Optimizer(abc.ABC):

    @abc.abstractmethod
    def optimize(self, model: Model, *args):
        pass


def all_equal(list: []):
    return len(list) == 0 or list.count(list[0]) == len(list)


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


class BatchedGradientOptimizer(Optimizer):

    def __init__(self, batch_size: int, epochs: int, shuffle=True):
        """
        :param epochs: number of epochs to train the model. Each epoch is a complete iteration over the training set.
        The number of parameter updates is n // batch_size, where n is the number of samples of the dataset
        :param batch_size: Batch the dataset with batches of size `batch_size`, and perform an optimization step
        for each batch
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle

    def backpropagation(self, model: Model, x: np.ndarray, y_true: np.ndarray, error_layer: Model):
        # forward pass (model and error)
        y = model.forward(x)
        E = error_layer.forward(y_true, y)

        # backward pass (error and model)
        dE_dy, _ = error_layer.backward(1)
        dE_dx, dE_dps = model.backward(dE_dy)

        return dE_dx, dE_dps, E

    def optimize(self, model: Model, x: np.ndarray, y: np.ndarray, error_layer: Model, verbose=True):
        """
        Fit a model to a dataset.
        :param model: the Model to optimize
        :param x: dataset inputs
        :param y: dataset outputs
        :param error_layer: To be applied to the output of the last layer
        :return:
        """
        n = x.shape[0]
        batches = n // self.batch_size
        history = []
        model.set_phase(Phase.Training)
        bar = tqdm(range(self.epochs), desc=f"optim. {model.name}", file=sys.stdout, disable=not verbose)
        for epoch in bar:
            epoch_error = 0
            for i, (x_batch, y_batch) in enumerate(batch_arrays(self.batch_size, x, y, shuffle=self.shuffle)):
                dE_dx, dE_dps, batch_error = self.backpropagation(model, x_batch, y_batch, error_layer)
                self.optimize_batch(model, dE_dps, epoch, i)
                epoch_error += batch_error
            epoch_error /= batches
            history.append(epoch_error)
            bar.set_postfix_str(f"{error_layer.name}: {epoch_error:.5f}")

        return np.array(history)

    @abc.abstractmethod
    def optimize_batch(self, model: Model, x: np.ndarray, y: np.ndarray, error_layer: Model, epoch: int):
        pass


class GradientDescent(BatchedGradientOptimizer):

    def __init__(self, batch_size: int, epochs: int, lr: float = 0.1, shuffle=True):
        super().__init__(batch_size, epochs, shuffle)
        self.lr = lr

    def optimize_batch(self, model: Model, dE_dps: ParameterSet, epoch: int, iteration: int):
        # Update parameters
        parameters = model.get_parameters()
        for parameter_name, dE_dp in dE_dps.items():
            p = parameters[parameter_name]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            """ YOUR IMPLEMENTATION START """
            p[:] = p - self.lr * dE_dp
            """ YOUR IMPLEMENTATION END """


class RMSprop(BatchedGradientOptimizer):

    def __init__(
        self, batch_size: int, epochs: int, lr: float = 0.1, beta: float = 0.99, eps: float = 1e-8, shuffle=True
    ):
        super().__init__(batch_size, epochs, shuffle)
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.first = True
        self.v = {}

    def optimize_batch(self, model: Model, dE_dps: ParameterSet, epoch: int, iteration: int):
        if self.first:
            self.first = False
            for k, p in model.get_parameters().items():
                self.v[k] = np.zeros_like(p)

        # Update parameters
        parameters = model.get_parameters()
        for parameter_name, dE_dp in dE_dps.items():
            p = parameters[parameter_name]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            """ YOUR IMPLEMENTATION START """
            self.v[parameter_name] = self.beta * self.v[parameter_name] + (1 - self.beta) * dE_dp * dE_dp
            p[:] = p - self.lr / (np.sqrt(self.v[parameter_name]) + self.eps) * dE_dp
            """ YOUR IMPLEMENTATION END """


class Adam(BatchedGradientOptimizer):

    def __init__(
        self, batch_size: int, epochs: int, lr: float = 0.1, betas: tuple = (0.9, 0.999), eps: int = 1e-08, shuffle=True
    ):
        super().__init__(batch_size, epochs, shuffle)
        self.lr = lr
        self.beta_1, self.beta_2 = betas
        self.eps = eps
        self.first = True
        self.m = {}
        self.v = {}

    def optimize_batch(self, model: Model, dE_dps: ParameterSet, epoch: int, iteration: int):
        if self.first:
            self.first = False
            for k, p in model.get_parameters().items():
                self.m[k] = np.zeros_like(p)
                self.v[k] = np.zeros_like(p)
        iteration += 1

        # Update parameters
        parameters = model.get_parameters()
        for parameter_name, dE_dp in dE_dps.items():
            p = parameters[parameter_name]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            """ YOUR IMPLEMENTATION START """
            self.m[parameter_name] = self.beta_1 * self.m[parameter_name] + (1 - self.beta_1) * dE_dp
            self.v[parameter_name] = self.beta_2 * self.v[parameter_name] + (1 - self.beta_2) * dE_dp * dE_dp
            m_hat = self.m[parameter_name] / (1 - np.power(self.beta_1, iteration))
            v_hat = self.v[parameter_name] / (1 - np.power(self.beta_2, iteration))
            p[:] = p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            """ YOUR IMPLEMENTATION END """


class MomentumGD(BatchedGradientOptimizer):

    def __init__(self, batch_size: int, epochs: int, lr: float = 0.1, gamma=0.9, shuffle=True):
        super().__init__(batch_size, epochs, shuffle)
        self.lr = lr
        self.gamma = gamma
        self.first = True
        self.v = {}

    def optimize_batch(self, model: Model, dE_dps: ParameterSet, epoch: int, iteration: int):
        if self.first:
            self.first = False
            for k, p in model.get_parameters().items():
                self.v[k] = np.zeros_like(p)

        # Update parameters
        parameters = model.get_parameters()
        for k, dE_dp in dE_dps.items():
            # K = parameter name
            p = parameters[k]
            v = self.v[k]
            # use p[:] and v[:] so that updates are in-place
            # instead of creating a new variable
            """ YOUR IMPLEMENTATION START """
            v[:] = self.gamma * v + self.lr * dE_dp
            p[:] = p - v
            """ YOUR IMPLEMENTATION END """


class NesterovMomentumGD(BatchedGradientOptimizer):

    def __init__(self, batch_size: int, epochs: int, lr: float = 0.1, gamma=0.9, shuffle=True):
        super().__init__(batch_size, epochs, shuffle)
        self.lr = lr
        self.gamma = gamma
        self.first = True
        self.v = {}

    def optimize_batch(self, model: Model, dE_dps: ParameterSet, epoch: int, iteration: int):
        if self.first:
            self.first = False
            for k, p in model.get_parameters().items():
                self.v[k] = np.zeros_like(p)

        # Update parameters
        parameters = model.get_parameters()
        for k, dE_dp in dE_dps.items():
            # K = parameter name
            p = parameters[k]
            v = self.v[k]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            """ YOUR IMPLEMENTATION START """
            v[:] = self.gamma * v + self.lr * dE_dp
            p[:] = p - (self.gamma * v + self.lr * dE_dp)
            """ YOUR IMPLEMENTATION END """


class SignGD(BatchedGradientOptimizer):

    def __init__(self, batch_size: int, epochs: int, lr: float = 0.1, eps=1e-8, shuffle=True):
        super().__init__(batch_size, epochs, shuffle)
        self.eps = eps
        self.lr = lr

    def optimize_batch(self, model: Model, dE_dps: ParameterSet, epoch: int, iteration: int):
        # Update parameters
        parameters = model.get_parameters()
        for parameter_name, dE_dp in dE_dps.items():
            p = parameters[parameter_name]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            """ YOUR IMPLEMENTATION START """
            denom = np.sqrt(dE_dp**2 + self.eps)
            p[:] = p - self.lr * (dE_dp / denom)
            """ YOUR IMPLEMENTATION END """
