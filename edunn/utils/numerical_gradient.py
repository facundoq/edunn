import numpy as np


def numerical_gradient(f, x: np.ndarray, δEδy: np.ndarray = None, h=1e-5):
    """
    Calculates the numerical gradient of E wrt x
    E is assumed to be a scalar, so that δEδy has size equal to y
    """

    def indices_generator(x):
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            # evaluate function at x+h
            ix = it.multi_index
            yield ix
            it.iternext()

    h2 = 2 * h
    δEδx = np.zeros_like(x)
    for i in indices_generator(x):
        oldval = x[i]
        # increment by h
        x[i] = oldval + h
        # evaluate f(x + h)
        fxph = f(x)
        # decrement by h2
        x[i] = oldval - h2
        # evaluate f(x - h)
        fxmh = f(x)
        # reset
        x[i] = oldval

        δyδxi = (fxph - fxmh) / h2

        δE = (δyδxi * δEδy)
        δE = δE.sum()
        δEδx[i] = δE
    return δEδx
