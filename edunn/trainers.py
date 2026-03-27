import numpy as np
from .model import Model, Phase
import sys
from tqdm.auto import tqdm
from .optimizer import Optimizer, batch_arrays

class SupervisedTrainer:
    def __init__(self, model: Model, optimizer: Optimizer, error_layer: Model, epochs: int, batch_size: int, verbose=True):
        self.model = model
        self.optimizer = optimizer
        self.error_layer = error_layer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def train(self, x: np.ndarray, y: np.ndarray, verbose=None):
        if verbose is None:
            verbose = self.verbose

        if hasattr(self.optimizer, 'epochs'):
            self.optimizer.epochs = self.epochs
        if hasattr(self.optimizer, 'batch_size'):
            self.optimizer.batch_size = self.batch_size

        return self.optimizer.optimize(self.model, x, y, self.error_layer, verbose=verbose)

class SequenceTrainer(SupervisedTrainer):
    def __init__(self, model: Model, optimizer: Optimizer, error_layer: Model, epochs: int, batch_size: int,
                 shuffle=True, accumulation_steps=1, verbose=True):
        super().__init__(model, optimizer, error_layer, epochs, batch_size, verbose)
        self.shuffle = shuffle
        self.accumulation_steps = accumulation_steps

    def train(self, x: np.ndarray, y: np.ndarray, verbose=None):
        if verbose is None:
            verbose = self.verbose

        n = x.shape[0]
        batches = max(1, n // self.batch_size)
        history = []
        self.model.set_phase(Phase.Training)

        bar = tqdm(range(self.epochs), desc=f"train {self.model.name}", file=sys.stdout, disable=not verbose)
        for epoch in bar:
            epoch_error = 0
            accumulated_gradients = {}
            steps = 0

            for i, (x_batch, y_batch) in enumerate(batch_arrays(self.batch_size, x, y, shuffle=self.shuffle)):
                y_pred = self.model.forward(x_batch)
                batch_error = self.error_layer.forward(y_batch, y_pred)
                epoch_error += batch_error

                δEδy, _ = self.error_layer.backward(1)
                _, δEδps = self.model.backward(δEδy)

                for name, grad in δEδps.items():
                    if name not in accumulated_gradients:
                        accumulated_gradients[name] = np.zeros_like(grad)
                    accumulated_gradients[name] += grad

                steps += 1
                if steps >= self.accumulation_steps:
                    for name in accumulated_gradients:
                        accumulated_gradients[name] /= steps

                    self.optimizer.optimize_batch(self.model, accumulated_gradients, epoch, i)
                    accumulated_gradients = {}
                    steps = 0

            if steps > 0:
                for name in accumulated_gradients:
                    accumulated_gradients[name] /= steps
                self.optimizer.optimize_batch(self.model, accumulated_gradients, epoch, i)

            epoch_error /= batches
            history.append(epoch_error)
            bar.set_postfix_str(f"{self.error_layer.name}: {epoch_error:.5f}")

        return np.array(history)
