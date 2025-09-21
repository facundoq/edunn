import numpy as np
from ..model import ModelWithParameters
from ..initializers import Initializer, RandomUniform


class RNN(ModelWithParameters):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        bptt_truncate: int = 4,
        kernel_initializer: Initializer = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        if kernel_initializer is None:
            kernel_initializer = RandomUniform()

        U = kernel_initializer.create((hidden_dim, input_dim))
        W = kernel_initializer.create((hidden_dim, hidden_dim))
        V = kernel_initializer.create((output_dim, hidden_dim))

        self.register_parameter("U", U)
        self.register_parameter("W", W)
        self.register_parameter("V", V)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch_size, timesteps, input_dim)
        return: (batch_size, timesteps, output_dim)
        """
        batch_size, timesteps, _ = x.shape
        U = self.get_parameters()["U"]
        W = self.get_parameters()["W"]
        V = self.get_parameters()["V"]

        h = np.zeros((batch_size, timesteps, self.hidden_dim))
        y = np.zeros((batch_size, timesteps, self.output_dim))
        h_prev = np.zeros((batch_size, self.hidden_dim))

        """YOUR IMPLEMENTATION START"""
        for t in range(timesteps):
            h_current = np.tanh(x[:, t, :] @ U.T + h_prev @ W.T)
            y[:, t, :] = h_current @ V.T
            h[:, t, :], h_prev = h_current, h_current
        """YOUR IMPLEMENTATION END"""

        self.set_cache(x, h)
        return y

    def backward(self, δEδy: np.ndarray) -> tuple:
        x, h = self.get_cache()
        batch_size, timesteps, _ = x.shape
        U = self.get_parameters()["U"]
        W = self.get_parameters()["W"]
        V = self.get_parameters()["V"]

        δEδU = np.zeros_like(U)
        δEδW = np.zeros_like(W)
        δEδV = np.zeros_like(V)
        δEδx = np.zeros_like(x)
        δEδh_next = np.zeros((batch_size, self.hidden_dim))

        """YOUR IMPLEMENTATION START"""
        # steps_back = min(self.bptt_truncate, t)
        for t in reversed(range(timesteps)):
            # (vocab_size, batch_size) x (batch_size, hidden_dim) = (vocab_size, hidden_dim)
            δEδV += δEδy[:, t, :].T @ h[:, t, :]
            # (batch_size, vocab_size) x (vocab_size, hidden_dim) + (batch_size, hidden_dim)
            δEδh = δEδy[:, t, :] @ V + δEδh_next

            δEδa = (1 - h[:, t, :] ** 2) * δEδh

            # (hidden_dim, batch_size) x (batch_size, vocab_size) = (hidden_dim, vocab_size)
            δEδU += δEδa.T @ x[:, t, :]

            # (batch_size, hidden_dim) x (hidden_dim, vocab_size) = (batch_size, vocab_size)
            δEδx[:, t, :] = δEδa @ U

            if t > 0:
                # (hidden_dim, batch_size) x (batch_size, hidden_dim) = (hidden_dim, hidden_dim)
                δEδW += δEδa.T @ h[:, t - 1, :]
                # (batch_size, hidden_dim) x (hidden_dim, hidden_dim) = (batch_size, hidden_dim)
                δEδh_next = δEδa @ W
            else:
                δEδh_next = np.zeros((batch_size, self.hidden_dim))
        """YOUR IMPLEMENTATION END"""

        # clip to mitigate exploding gradients
        for grad in [δEδU, δEδW, δEδV]:
            np.clip(grad, -1e5, 1e5, out=grad)

        return δEδx, {"U": δEδU, "W": δEδW, "V": δEδV}
