import numpy as np


def numerical_gradient(f, x: np.ndarray, dE_dy: np.ndarray = None, h=1e-5) -> np.ndarray:
    """
    Calculates the numerical gradient of E wrt x
    E is assumed to be a scalar, so that dE_dy has size equal to y
    """

    def indices_generator(x):
        it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            # evaluate function at x+h
            ix = it.multi_index
            yield ix
            it.iternext()

    h2 = 2 * h
    dE_dx = np.zeros_like(x)
    for i in indices_generator(x):
        x_value = x[i]
        # increment by h
        x[i] = x_value + h
        # evaluate f(x + h)
        fxph = f(x)
        # decrement by h
        x[i] = x_value - h
        # evaluate f(x - h)
        fxmh = f(x)
        # reset
        x[i] = x_value

        dy_dxi = (fxph - fxmh) / h2

        dE = dy_dxi * dE_dy
        dE = dE.sum()
        dE_dx[i] = dE
    return dE_dx
