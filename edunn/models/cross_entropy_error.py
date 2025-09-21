import numpy as np

from ..model import ModelWithoutParameters


class CrossEntropyWithLabels(ModelWithoutParameters):
    """
    Returns the CrossEntropy between two class distributions
    Receives a matrix of probabilities NxC for each sample and a vector
    of class labels, also one for each sample,.
    For each sample, this layer receives a vector of probabilities for each class,
    (which must sum to 1)
    and the label of the sample (which implies that class has an
    expected probability of 1, and 0 for the rest)
    """

    # Ayuda para implementar:
    # http://facundoq.github.io/edunn/material/crossentropy_derivative.html
    def forward(self, y_true: np.ndarray, y: np.ndarray):
        y_true = np.squeeze(y_true)
        assert len(y_true.shape) == 1
        assert y.min() >= 0
        n, c = y.shape

        E = np.zeros((n, 1))
        """YOUR IMPLEMENTATION START"""
        for i in range(n):
            probability = y[i, y_true[i]]
            E[i] = -np.log(probability)
        """YOUR IMPLEMENTATION END"""
        assert np.all(np.squeeze(E).shape == y_true.shape)
        self.set_cache(y_true, y)
        return E

    def backward(self, δEδyi):
        y_true, y = self.get_cache()

        δEδy = np.zeros_like(y)
        n, classes = y.shape
        """YOUR IMPLEMENTATION START"""
        for i in range(n):
            p = y[i, y_true[i]]
            δEδy[i, y_true[i]] = -1 / p
        """YOUR IMPLEMENTATION END"""
        return δEδy * δEδyi, {}


def log_softmax(x):
    lse = np.zeros_like(x)
    x_max = np.max(x, axis=-1, keepdims=True)
    """YOUR IMPLEMENTATION START"""
    x_adj = x - x_max
    lse = x_adj - np.log(np.sum(np.exp(x_adj), axis=-1, keepdims=True))
    """YOUR IMPLEMENTATION END"""
    return lse


class SequenceCrossEntropyWithLabels(ModelWithoutParameters):
    def __init__(self, pad_index=0):
        super().__init__()
        self.pad_index = pad_index

    def forward(self, y_true: np.ndarray, y: np.ndarray):
        """
        y_true: (batch_size, timesteps, vocab_size) one-hot
        y:      (batch_size, timesteps, vocab_size) logits
        """
        E = 0

        """YOUR IMPLEMENTATION START"""
        # mask: 1 if token isn't padding
        self.mask = 1 - y_true[..., self.pad_index]  # (B, T)
        self.valid_tokens = np.sum(self.mask)  # (1,)

        # log-softmax for stability
        log_probs = log_softmax(y)  # (B, T, C)

        # loss: one-hot * log_probs, sum over classes
        true_log = np.sum(y_true * log_probs, axis=-1)  # (B, T)

        # total loss
        total_loss = -np.sum(true_log * self.mask)

        # mean over tokens
        E = total_loss / max(self.valid_tokens, 1)
        """YOUR IMPLEMENTATION END"""

        self.set_cache(y_true, y)
        return E

    def backward(self, δEδyi):
        y_true, y = self.get_cache()  # (B, T, C)

        δEδy = np.zeros_like(y)

        """YOUR IMPLEMENTATION START"""
        # log-softmax -> softmax probabilities
        log_probs = log_softmax(y)
        probs = np.exp(log_probs)  # (B, T, C)

        # grad softmax−crossentropy: probs - one-hot
        δEδy = probs - y_true  # (B, T, C)

        # mask to ignore padding, normalization
        δEδy *= self.mask[..., None]
        δEδy /= max(self.valid_tokens, 1)
        """YOUR IMPLEMENTATION END"""

        return δEδy * δEδyi, None
