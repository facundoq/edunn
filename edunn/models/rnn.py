import numpy as np
from ..model import Model
from ..initializers import Initializer, RandomUniform


class RNN(Model):
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

        """ YOUR IMPLEMENTATION START """
        for t in range(timesteps):
            h_current = np.tanh(x[:, t, :] @ U.T + h_prev @ W.T)
            y[:, t, :] = h_current @ V.T
            h[:, t, :], h_prev = h_current, h_current
        """ YOUR IMPLEMENTATION END """

        self.set_cache(x, h)
        return y

    def backward(self, dE_dy: np.ndarray) -> tuple:
        x, h = self.get_cache()
        batch_size, timesteps, _ = x.shape
        U = self.get_parameters()["U"]
        W = self.get_parameters()["W"]
        V = self.get_parameters()["V"]

        dE_dU = np.zeros_like(U)
        dE_dW = np.zeros_like(W)
        dE_dV = np.zeros_like(V)
        dE_dx = np.zeros_like(x)
        dh_next = np.zeros((batch_size, self.hidden_dim))

        """ YOUR IMPLEMENTATION START """
        # steps_back = min(self.bptt_truncate, t)
        for t in reversed(range(timesteps)):
            # (vocab_size, batch_size) x (batch_size, hidden_dim) = (vocab_size, hidden_dim)
            dE_dV += dE_dy[:, t, :].T @ h[:, t, :]
            # (batch_size, vocab_size) x (vocab_size, hidden_dim) + (batch_size, hidden_dim)
            dE_dh = dE_dy[:, t, :] @ V + dh_next

            dh_raw = (1 - h[:, t, :] ** 2) * dE_dh

            # (hidden_dim, batch_size) x (batch_size, vocab_size) = (hidden_dim, vocab_size)
            dE_dU += dh_raw.T @ x[:, t, :]

            # (batch_size, hidden_dim) x (hidden_dim, vocab_size) = (batch_size, vocab_size)
            dE_dx[:, t, :] = dh_raw @ U

            if t > 0:
                # (hidden_dim, batch_size) x (batch_size, hidden_dim) = (hidden_dim, hidden_dim)
                dE_dW += dh_raw.T @ h[:, t - 1, :]
                # (batch_size, hidden_dim) x (hidden_dim, hidden_dim) = (batch_size, hidden_dim)
                dh_next = dh_raw @ W
            else:
                dh_next = np.zeros((batch_size, self.hidden_dim))
        """ YOUR IMPLEMENTATION END """

        # clip to mitigate exploding gradients
        for grad in [dE_dU, dE_dW, dE_dV]:
            np.clip(grad, -1e5, 1e5, out=grad)

        return dE_dx, {"U": dE_dU, "W": dE_dW, "V": dE_dV}
