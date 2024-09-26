import numpy as np
from ..model import Model


def conv2d_forward(x, func, stride=(1, 1), pool_size=(1, 1)):
    stride_h, stride_w = stride

    bx, cx, hx, wx = x.shape
    ph, pw = pool_size

    hy = int((hx - ph) / stride_h) + 1
    wy = int((wx - pw) / stride_w) + 1

    y = np.zeros((bx, cx, hy, wy))

    """ YOUR IMPLEMENTATION START """
    for i in range(hy):
        for j in range(wy):
            for k in range(cx):
                for l in range(bx):
                    y[l, k, i, j] = func(
                        x[l, k, i * stride_h: i * stride_h + ph, j * stride_w: j * stride_w + pw])
    """ YOUR IMPLEMENTATION END """

    return y


def conv2d_backward_max(dy, x, stride=(1, 1), pool_size=(1, 1)):
    stride_h, stride_w = stride

    bx, cx, hx, wx = x.shape
    ph, pw = pool_size

    hy = int((hx - ph) / stride_h) + 1
    wy = int((wx - pw) / stride_w) + 1

    dx = np.zeros_like(x)

    """ YOUR IMPLEMENTATION START """
    for i in range(hy):
        for j in range(wy):
            for k in range(cx):
                for l in range(bx):
                    # get the index in the region i,j where the value is the maximum
                    i_t, j_t = np.where(np.max(
                        x[l, k, i * stride_h: i * stride_h + ph, j * stride_w: j * stride_w + pw]) ==
                                        x[l, k, i * stride_h: i * stride_h + ph, j * stride_w: j * stride_w + pw])
                    i_t, j_t = i_t[0], j_t[0]
                    # only the position of the maximum element in the region i,j gets the incoming gradient, the other gradients are zero
                    dx[l, k, i * stride_h: i * stride_h + ph, j * stride_w: j * stride_w + pw][i_t, j_t] = dy[
                        l, k, i, j]
    """ YOUR IMPLEMENTATION END """

    return dx


def conv2d_backward_avg(dy, x, stride=(1, 1), pool_size=(1, 1)):
    stride_h, stride_w = stride

    bx, cx, hx, wx = x.shape
    ph, pw = pool_size

    hy = int((hx - ph) / stride_h) + 1
    wy = int((wx - pw) / stride_w) + 1

    dx = np.zeros_like(x)

    """ YOUR IMPLEMENTATION START """
    for i in range(hy):
        for j in range(wy):
            for k in range(cx):
                for l in range(bx):
                    dy_avg = dy[l, k, i, j] / (ph * pw)
                    dx[l, k, i * stride_h:(i * stride_h + ph), j * stride_w:(j * stride_w + pw)] += np.ones(
                        (ph, pw)) * dy_avg
    """ YOUR IMPLEMENTATION END """

    return dx


class MaxPool2d(Model):

    def __init__(self, kernel_size: int, stride: int = 1, name=None):
        super().__init__(name=name)
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

    def forward(self, x: np.ndarray):
        y = {}
        """ YOUR IMPLEMENTATION START """
        y = conv2d_forward(x, np.max, self.stride, self.kernel_size)
        """ YOUR IMPLEMENTATION END """
        self.set_cache(x)
        return y

    def backward(self, dE_dy: np.ndarray):
        dE_dx = {}
        x, = self.get_cache()
        """ YOUR IMPLEMENTATION START """
        dE_dx = conv2d_backward_max(dE_dy, x, self.stride, self.kernel_size)
        """ YOUR IMPLEMENTATION END """
        return dE_dx, {}


class AvgPool2d(Model):

    def __init__(self, kernel_size: int, stride: int = 1, name=None):
        super().__init__(name=name)
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

    def forward(self, x: np.ndarray):
        y = {}
        self.set_cache(x)
        """ YOUR IMPLEMENTATION START """
        y = conv2d_forward(x, np.average, self.stride, self.kernel_size)
        """ YOUR IMPLEMENTATION END """
        return y

    def backward(self, dE_dy: np.ndarray):
        dE_dx = {}
        x, = self.get_cache()
        """ YOUR IMPLEMENTATION START """
        dE_dx = conv2d_backward_avg(dE_dy, x, self.stride, self.kernel_size)
        """ YOUR IMPLEMENTATION END """
        return dE_dx, {}
