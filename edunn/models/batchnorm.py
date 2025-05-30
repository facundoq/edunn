import numpy as np
from ..model import Model
from ..initializers import Initializer, RandomNormal


class BatchNorm(Model):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momemtum: float = 0.1,
        affine: bool = True,
        gamma_initializer: Initializer = None,
        beta_initializer: Initializer = None,
        name=None,
    ):
        super().__init__(name=name)
        self.num_features = num_features
        self.eps = eps
        if gamma_initializer is None:
            gamma_initializer = RandomNormal()
        if beta_initializer is None:
            beta_initializer = RandomNormal()
        w = gamma_initializer.create(num_features)
        b = beta_initializer.create(num_features)
        self.register_parameter("w", w)
        self.register_parameter("b", b)
        # self.bias = Bias(num_features, initializer=beta_initializer)

    def forward(self, x: np.ndarray):
        y = {}
        cache = tuple()

        # Retrieve w
        w = self.get_parameters()["w"]
        b = self.get_parameters()["b"]

        """ YOUR IMPLEMENTATION START """
        x_mean = x.mean(axis=0)
        x_var = x.var(axis=0)

        std = np.sqrt(x_var + self.eps)
        x_norm = (x - x_mean) / std
        y = w * x_norm + b

        cache = (x_norm, std, w)
        """ YOUR IMPLEMENTATION END """

        self.set_cache(cache)
        return y

    def backward(self, dE_dy: np.ndarray):
        dE_dx, dE_dw, dE_db = {}, {}, {}
        # Retrieve variables from cache
        ((x_norm, std, gamma),) = self.get_cache()
        """ YOUR IMPLEMENTATION START """
        N = dE_dy.shape[0]

        dE_dw = (dE_dy * x_norm).sum(axis=0)
        dE_db = dE_dy.sum(axis=0)

        dx_norm = dE_dy * gamma
        dE_dx = 1 / N / std * (N * dx_norm - dx_norm.sum(axis=0) - x_norm * (dx_norm * x_norm).sum(axis=0))
        """ YOUR IMPLEMENTATION END """
        dE_dbn = {"w": dE_dw, "b": dE_db}
        return dE_dx, dE_dbn
