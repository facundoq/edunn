import numpy as np

from ..model import Model


class CrossEntropyWithLabels(Model):
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
    # https://facundoq.github.io/edunn/material/crossentropy_derivative.html
    def forward(self, y_true: np.ndarray, y: np.ndarray):
        y_true = np.squeeze(y_true)
        assert len(y_true.shape) == 1
        assert y.min() >= 0
        n, c = y.shape

        E = np.zeros((n, 1))
        """ YOUR IMPLEMENTATION START """
        for i in range(n):
            probability = y[i, y_true[i]]
            E[i] = -np.log(probability)
        """ YOUR IMPLEMENTATION END """
        assert np.all(np.squeeze(E).shape == y_true.shape)
        self.set_cache(y_true, y)
        return E

    def backward(self, dE_dyi):
        y_true, y = self.get_cache()

        dE_dy = np.zeros_like(y)
        n, classes = y.shape
        """ YOUR IMPLEMENTATION START """
        for i in range(n):
            p = y[i, y_true[i]]
            dE_dy[i, y_true[i]] = -1 / p
        """ YOUR IMPLEMENTATION END """
        return dE_dy * dE_dyi, {}
