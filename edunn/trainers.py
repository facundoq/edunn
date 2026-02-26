# Training strategies for eduNN models.
#
# A Trainer handles the training loop: batching, epochs, forward/backward passes,
# and calling the Optimizer to update parameters. Different Trainer subclasses
# implement different training strategies (supervised, recurrent, etc.).
#
# For backward compatibility, this module also provides wrapper classes that
# combine an Optimizer with a Trainer under the old API
# (e.g., GradientDescent(batch_size, epochs, lr).optimize(...)).


from typing import Dict
import numpy as np
from .model import Model, Phase
from .model import ParameterSet
from .optimizer import (
    Optimizer,
    SGD,
    MomentumSGD,
    NesterovMomentumSGD,
    RMSpropOptimizer,
    AdamOptimizer,
    SignSGDOptimizer,
)
import sys
import abc
import random
from tqdm.auto import tqdm


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


def _accumulate_gradients(accumulated: ParameterSet, new_grads: ParameterSet):
    """Add new_grads into accumulated in-place. If accumulated is empty, initialize it."""
    for k, v in new_grads.items():
        if k in accumulated:
            accumulated[k] = accumulated[k] + v
        else:
            accumulated[k] = v.copy()


def _scale_gradients(gradients: ParameterSet, scale: float) -> ParameterSet:
    """Scale all gradients by a constant factor."""
    return {k: v * scale for k, v in gradients.items()}


# ---------------------------------------------------------------------------
# Trainer base class and strategies
# ---------------------------------------------------------------------------


class Trainer(abc.ABC):
    """
    Base class for training strategies.

    A Trainer orchestrates the training loop: iterating over epochs and batches,
    performing forward/backward passes, and calling an Optimizer to update
    model parameters.

    Subclasses implement different training strategies for different model types
    (feedforward, recurrent, generative, etc.).
    """

    def __init__(self, optimizer: Optimizer, batch_size: int, epochs: int, shuffle: bool = True):
        """
        :param optimizer: the Optimizer algorithm to use for parameter updates
        :param batch_size: size of mini-batches
        :param epochs: number of epochs (full passes over the training set)
        :param shuffle: whether to shuffle batches each epoch
        """
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle

    def backpropagation(self, model: Model, x: np.ndarray, y_true: np.ndarray, error_layer: Model):
        """
        Perform a single forward + backward pass.

        :param model: the model
        :param x: input batch
        :param y_true: target batch
        :param error_layer: error/loss layer
        :return: (input_gradient, parameter_gradients, error_value)
        """
        # forward pass (model and error)
        y = model.forward(x)
        E = error_layer.forward(y_true, y)

        # backward pass (error and model)
        δEδy, _ = error_layer.backward(1)
        δEδx, δEδps = model.backward(δEδy)

        return δEδx, δEδps, E

    @abc.abstractmethod
    def train(self, model: Model, x: np.ndarray, y: np.ndarray, error_layer: Model, verbose=True) -> np.ndarray:
        """
        Train a model on a dataset.

        :param model: the Model to train
        :param x: dataset inputs
        :param y: dataset outputs/targets
        :param error_layer: loss function layer
        :param verbose: whether to show progress
        :return: array of per-epoch error values (training history)
        """
        pass

    def optimize(self, model: Model, x: np.ndarray, y: np.ndarray, error_layer: Model, verbose=True) -> np.ndarray:
        """
        Alias for train(), provided for backward compatibility.
        """
        return self.train(model, x, y, error_layer, verbose)


class SupervisedTrainer(Trainer):
    """
    Standard supervised training strategy.

    Performs forward/backward on each mini-batch, then immediately updates
    parameters via the optimizer. This is the typical training loop for
    feedforward networks (MLPs, CNNs, etc.).
    """

    def __init__(self, optimizer: Optimizer, batch_size: int, epochs: int, shuffle: bool = True):
        super().__init__(optimizer, batch_size, epochs, shuffle)

    def train(self, model: Model, x: np.ndarray, y: np.ndarray, error_layer: Model, verbose=True) -> np.ndarray:
        """
        Fit a model to a dataset using standard supervised training.

        :param model: the Model to optimize
        :param x: dataset inputs
        :param y: dataset outputs
        :param error_layer: To be applied to the output of the last layer
        :return: array of per-epoch error values
        """
        n = x.shape[0]
        batches = n // self.batch_size
        history = []
        model.set_phase(Phase.Training)
        self.optimizer.initialize(model.get_parameters())
        bar = tqdm(range(self.epochs), desc=f"optim. {model.name}", file=sys.stdout, disable=not verbose)
        for epoch in bar:
            epoch_error = 0
            for i, (x_batch, y_batch) in enumerate(batch_arrays(self.batch_size, x, y, shuffle=self.shuffle)):
                δEδx, δEδps, batch_error = self.backpropagation(model, x_batch, y_batch, error_layer)
                self.optimizer.step(model.get_parameters(), δEδps, epoch, i)
                epoch_error += batch_error
            epoch_error /= batches
            history.append(epoch_error)
            bar.set_postfix_str(f"{error_layer.name}: {epoch_error:.5f}")

        return np.array(history)


class RecurrentTrainer(Trainer):
    """
    Training strategy for recurrent models (RNNs, LSTMs, etc.).

    Unlike SupervisedTrainer which updates parameters after every batch,
    RecurrentTrainer supports gradient accumulation across multiple batches
    before performing a parameter update. This is useful for:
    - Training on long sequences where memory is limited
    - Simulating larger effective batch sizes
    - Stabilizing training of recurrent models

    The gradient_accumulation_steps parameter controls how many batches of
    gradients are accumulated before a single optimizer step.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        batch_size: int,
        epochs: int,
        shuffle: bool = True,
        gradient_accumulation_steps: int = 1,
    ):
        """
        :param optimizer: the Optimizer algorithm to use for parameter updates
        :param batch_size: size of mini-batches
        :param epochs: number of epochs
        :param shuffle: whether to shuffle batches each epoch
        :param gradient_accumulation_steps: number of batches over which to
            accumulate gradients before performing an optimizer step. A value
            of 1 means update every batch (same as SupervisedTrainer). A value
            of N means accumulate gradients over N batches, then update once
            with the averaged gradients.
        """
        super().__init__(optimizer, batch_size, epochs, shuffle)
        assert gradient_accumulation_steps >= 1, "gradient_accumulation_steps must be >= 1"
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def train(self, model: Model, x: np.ndarray, y: np.ndarray, error_layer: Model, verbose=True) -> np.ndarray:
        """
        Train a recurrent model with gradient accumulation.

        :param model: the Model to train
        :param x: dataset inputs (typically shape: batch x timesteps x features)
        :param y: dataset targets
        :param error_layer: loss function layer
        :param verbose: whether to show progress
        :return: array of per-epoch error values
        """
        n = x.shape[0]
        batches = n // self.batch_size
        history = []
        model.set_phase(Phase.Training)
        self.optimizer.initialize(model.get_parameters())
        bar = tqdm(range(self.epochs), desc=f"optim. {model.name}", file=sys.stdout, disable=not verbose)
        for epoch in bar:
            epoch_error = 0
            accumulated_grads = {}
            steps_since_update = 0

            for i, (x_batch, y_batch) in enumerate(batch_arrays(self.batch_size, x, y, shuffle=self.shuffle)):
                δEδx, δEδps, batch_error = self.backpropagation(model, x_batch, y_batch, error_layer)
                _accumulate_gradients(accumulated_grads, δEδps)
                steps_since_update += 1
                epoch_error += batch_error

                if steps_since_update >= self.gradient_accumulation_steps:
                    # Average the accumulated gradients and apply update
                    avg_grads = _scale_gradients(accumulated_grads, 1.0 / steps_since_update)
                    self.optimizer.step(model.get_parameters(), avg_grads, epoch, i)
                    accumulated_grads = {}
                    steps_since_update = 0

            # Handle any remaining accumulated gradients at end of epoch
            if steps_since_update > 0:
                avg_grads = _scale_gradients(accumulated_grads, 1.0 / steps_since_update)
                self.optimizer.step(model.get_parameters(), avg_grads, epoch, batches)

            epoch_error /= batches
            history.append(epoch_error)
            bar.set_postfix_str(f"{error_layer.name}: {epoch_error:.5f}")

        return np.array(history)


# ---------------------------------------------------------------------------
# Backward-compatible wrapper classes
# ---------------------------------------------------------------------------
#
# These classes preserve the old API where the optimizer algorithm and
# training loop were combined into a single object:
#
#     optimizer = nn.GradientDescent(batch_size=32, epochs=100, lr=0.1)
#     history = optimizer.optimize(model, x, y, error)
#
# Internally they delegate to the new separated Optimizer + SupervisedTrainer.
# ---------------------------------------------------------------------------


class BatchedGradientOptimizer(SupervisedTrainer):
    """
    Base class for backward-compatible optimizer+trainer wrappers.

    This class exists to preserve the old API where optimizer algorithm
    and training strategy were combined. New code should use
    SupervisedTrainer (or RecurrentTrainer) with a separate Optimizer instead.
    """
    pass


class GradientDescent(BatchedGradientOptimizer):
    """
    Backward-compatible wrapper combining SGD optimizer with supervised training.

    Old API (still works):
        optimizer = nn.GradientDescent(batch_size=32, epochs=100, lr=0.1)
        history = optimizer.optimize(model, x, y, error)

    New equivalent:
        optimizer = nn.SGD(lr=0.1)
        trainer = nn.SupervisedTrainer(optimizer, batch_size=32, epochs=100)
        history = trainer.train(model, x, y, error)
    """

    def __init__(self, batch_size: int, epochs: int, lr: float = 0.1, shuffle=True):
        super().__init__(SGD(lr), batch_size, epochs, shuffle)
        self.lr = lr

    def optimize_batch(self, model: Model, δEδps: ParameterSet, epoch: int, iteration: int):
        self.optimizer.step(model.get_parameters(), δEδps, epoch, iteration)


class RMSprop(BatchedGradientOptimizer):
    """
    Backward-compatible wrapper combining RMSprop optimizer with supervised training.

    Old API (still works):
        optimizer = nn.RMSprop(batch_size=32, epochs=100, lr=0.1)
        history = optimizer.optimize(model, x, y, error)

    New equivalent:
        optimizer = nn.RMSpropOptimizer(lr=0.1)
        trainer = nn.SupervisedTrainer(optimizer, batch_size=32, epochs=100)
        history = trainer.train(model, x, y, error)
    """

    def __init__(
        self, batch_size: int, epochs: int, lr: float = 0.1, beta: float = 0.99, eps: float = 1e-8, shuffle=True
    ):
        super().__init__(RMSpropOptimizer(lr, beta, eps), batch_size, epochs, shuffle)
        self.lr = lr

    def optimize_batch(self, model: Model, δEδps: ParameterSet, epoch: int, iteration: int):
        self.optimizer.step(model.get_parameters(), δEδps, epoch, iteration)


class Adam(BatchedGradientOptimizer):
    """
    Backward-compatible wrapper combining Adam optimizer with supervised training.

    Old API (still works):
        optimizer = nn.Adam(batch_size=32, epochs=100, lr=0.1)
        history = optimizer.optimize(model, x, y, error)

    New equivalent:
        optimizer = nn.AdamOptimizer(lr=0.1)
        trainer = nn.SupervisedTrainer(optimizer, batch_size=32, epochs=100)
        history = trainer.train(model, x, y, error)
    """

    def __init__(
        self, batch_size: int, epochs: int, lr: float = 0.1, betas: tuple = (0.9, 0.999), eps: int = 1e-08, shuffle=True
    ):
        super().__init__(AdamOptimizer(lr, betas, eps), batch_size, epochs, shuffle)
        self.lr = lr

    def optimize_batch(self, model: Model, δEδps: ParameterSet, epoch: int, iteration: int):
        self.optimizer.step(model.get_parameters(), δEδps, epoch, iteration)


class MomentumGD(BatchedGradientOptimizer):
    """
    Backward-compatible wrapper combining Momentum SGD optimizer with supervised training.

    Old API (still works):
        optimizer = nn.MomentumGD(batch_size=32, epochs=100, lr=0.1)
        history = optimizer.optimize(model, x, y, error)

    New equivalent:
        optimizer = nn.MomentumSGD(lr=0.1, gamma=0.9)
        trainer = nn.SupervisedTrainer(optimizer, batch_size=32, epochs=100)
        history = trainer.train(model, x, y, error)
    """

    def __init__(self, batch_size: int, epochs: int, lr: float = 0.1, gamma=0.9, shuffle=True):
        super().__init__(MomentumSGD(lr, gamma), batch_size, epochs, shuffle)
        self.lr = lr

    def optimize_batch(self, model: Model, δEδps: ParameterSet, epoch: int, iteration: int):
        self.optimizer.step(model.get_parameters(), δEδps, epoch, iteration)


class NesterovMomentumGD(BatchedGradientOptimizer):
    """
    Backward-compatible wrapper combining Nesterov Momentum SGD optimizer with supervised training.

    Old API (still works):
        optimizer = nn.NesterovMomentumGD(batch_size=32, epochs=100, lr=0.1)
        history = optimizer.optimize(model, x, y, error)

    New equivalent:
        optimizer = nn.NesterovMomentumSGD(lr=0.1, gamma=0.9)
        trainer = nn.SupervisedTrainer(optimizer, batch_size=32, epochs=100)
        history = trainer.train(model, x, y, error)
    """

    def __init__(self, batch_size: int, epochs: int, lr: float = 0.1, gamma=0.9, shuffle=True):
        super().__init__(NesterovMomentumSGD(lr, gamma), batch_size, epochs, shuffle)
        self.lr = lr

    def optimize_batch(self, model: Model, δEδps: ParameterSet, epoch: int, iteration: int):
        self.optimizer.step(model.get_parameters(), δEδps, epoch, iteration)


class SignGD(BatchedGradientOptimizer):
    """
    Backward-compatible wrapper combining SignSGD optimizer with supervised training.

    Old API (still works):
        optimizer = nn.SignGD(batch_size=32, epochs=100, lr=0.1)
        history = optimizer.optimize(model, x, y, error)

    New equivalent:
        optimizer = nn.SignSGDOptimizer(lr=0.1)
        trainer = nn.SupervisedTrainer(optimizer, batch_size=32, epochs=100)
        history = trainer.train(model, x, y, error)
    """

    def __init__(self, batch_size: int, epochs: int, lr: float = 0.1, eps=1e-8, shuffle=True):
        super().__init__(SignSGDOptimizer(lr, eps), batch_size, epochs, shuffle)
        self.lr = lr

    def optimize_batch(self, model: Model, δEδps: ParameterSet, epoch: int, iteration: int):
        self.optimizer.step(model.get_parameters(), δEδps, epoch, iteration)
