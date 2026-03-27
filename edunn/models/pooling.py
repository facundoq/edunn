import numpy as np
from ..model import ModelWithoutParameters


def conv2d_forward(x, func, stride=(1, 1), pool_size=(1, 1)):
    stride_h, stride_w = stride

    bx, cx, hx, wx = x.shape
    ph, pw = pool_size

    hy = int((hx - ph) / stride_h) + 1
    wy = int((wx - pw) / stride_w) + 1

    y = np.zeros((bx, cx, hy, wy))

    """YOUR IMPLEMENTATION START"""
    for i in range(hy):
        for j in range(wy):
            y[:, :, i, j] = func(x[:, :, i * stride_h : i * stride_h + ph, j * stride_w : j * stride_w + pw], axis=(2, 3))
    """YOUR IMPLEMENTATION END"""

    return y


def conv2d_backward_max(dy, x, stride=(1, 1), pool_size=(1, 1)):
    stride_h, stride_w = stride

    bx, cx, hx, wx = x.shape
    ph, pw = pool_size

    hy = int((hx - ph) / stride_h) + 1
    wy = int((wx - pw) / stride_w) + 1

    dx = np.zeros_like(x)

    """YOUR IMPLEMENTATION START"""
    for i in range(hy):
        for j in range(wy):
            x_slice = x[:, :, i * stride_h : i * stride_h + ph, j * stride_w : j * stride_w + pw]
            max_val = np.max(x_slice, axis=(2, 3), keepdims=True)
            mask = (x_slice == max_val)
            # If there are multiple maximums, divide the gradient equally
            mask_sum = np.sum(mask, axis=(2, 3), keepdims=True)
            dx[:, :, i * stride_h : i * stride_h + ph, j * stride_w : j * stride_w + pw] += mask * (dy[:, :, i, j][:, :, None, None] / mask_sum)
    """YOUR IMPLEMENTATION END"""

    return dx


def conv2d_backward_avg(dy, x, stride=(1, 1), pool_size=(1, 1)):
    stride_h, stride_w = stride

    bx, cx, hx, wx = x.shape
    ph, pw = pool_size

    hy = int((hx - ph) / stride_h) + 1
    wy = int((wx - pw) / stride_w) + 1

    dx = np.zeros_like(x)

    """YOUR IMPLEMENTATION START"""
    for i in range(hy):
        for j in range(wy):
            dy_avg = dy[:, :, i, j] / (ph * pw)
            dx[:, :, i * stride_h : i * stride_h + ph, j * stride_w : j * stride_w + pw] += dy_avg[:, :, None, None]
    """YOUR IMPLEMENTATION END"""

    return dx


class MaxPool2d(ModelWithoutParameters):

    def __init__(self, kernel_size: int, stride: int = 1, name=None):
        super().__init__(name=name)
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

    def forward(self, x: np.ndarray):
        y = {}
        """YOUR IMPLEMENTATION START"""
        y = conv2d_forward(x, np.max, self.stride, self.kernel_size)
        """YOUR IMPLEMENTATION END"""
        self.set_cache(x)
        return y

    def backward(self, δEδy: np.ndarray):
        δEδx = {}
        (x,) = self.get_cache()
        """YOUR IMPLEMENTATION START"""
        δEδx = conv2d_backward_max(δEδy, x, self.stride, self.kernel_size)
        """YOUR IMPLEMENTATION END"""
        return δEδx, {}


class AvgPool2d(ModelWithoutParameters):

    def __init__(self, kernel_size: int, stride: int = 1, name=None):
        super().__init__(name=name)
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

    def forward(self, x: np.ndarray):
        y = {}
        self.set_cache(x)
        """YOUR IMPLEMENTATION START"""
        y = conv2d_forward(x, np.average, self.stride, self.kernel_size)
        """YOUR IMPLEMENTATION END"""
        return y

    def backward(self, δEδy: np.ndarray):
        δEδx = {}
        (x,) = self.get_cache()
        """YOUR IMPLEMENTATION START"""
        δEδx = conv2d_backward_avg(δEδy, x, self.stride, self.kernel_size)
        """YOUR IMPLEMENTATION END"""
        return δEδx, {}
