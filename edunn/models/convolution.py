import numpy as np
from ..model import ModelWithParameters

from ..initializers import Initializer, RandomNormal

from .bias import Bias


def dilate2d(x, dilation):
    b, c, h, w = x.shape
    dilation_h, dilation_w = dilation
    new_shape = (b, c, h + (h - 1) * (dilation_h - 1), w + (w - 1) * (dilation_w - 1))
    x_dilated = np.zeros(new_shape)

    """YOUR IMPLEMENTATION START"""
    x_dilated[:, :, ::dilation_h, ::dilation_w] = x
    """YOUR IMPLEMENTATION END"""

    return x_dilated


def pad2d(x, pad_size):
    b, c, h, w = x.shape
    pad_size_h, pad_size_w = pad_size
    new_shape = (b, c, h + 2 * pad_size_h, w + 2 * pad_size_w)
    x_padded = np.zeros(new_shape)

    """YOUR IMPLEMENTATION START"""
    x_padded[:, :, pad_size_h : -pad_size_h if pad_size_h > 0 else h, pad_size_w:-pad_size_w] = x
    """YOUR IMPLEMENTATION END"""

    return x_padded


def is_odd(x):
    return x % 2 == 1


def conv2d_forward(w, x, strides=(1, 1), pad_size=(0, 0)):
    # Pad the input X before doing the convolution
    """YOUR IMPLEMENTATION START"""
    if pad_size[-1] > 0:
        x = pad2d(x, pad_size)
    """YOUR IMPLEMENTATION END"""

    stride_h, stride_w = strides

    bx, cx, hx, wx = x.shape
    bw, cw, hw, ww = w.shape
    assert is_odd(hw) and is_odd(ww), "The dimensions of w must be odd numbers"
    assert cx == cw, "The number of channels in the weight matrix must be equal to the number of channels in the image"

    hy = int((hx - hw) / stride_h) + 1
    wy = int((wx - ww) / stride_w) + 1
    y = np.zeros((bx, bw, hy, wy))

    # Compute the convolution between X and W to get Y
    # Hint: use multiple for loops for the expected size
    """YOUR IMPLEMENTATION START"""
    # Optimized: Loop over the kernel size instead of output spatial dimensions
    for a in range(hw):
        for b in range(ww):
            # x_slice has shape (bx, cx, hy, wy)
            x_slice = x[:, :, a:a + hy * stride_h:stride_h, b:b + wy * stride_w:stride_w]
            # y += einsum('oc...,bc...->bo...', w[:, :, a, b], x_slice)
            y += np.einsum("oc,bcHW->boHW", w[:, :, a, b], x_slice)
    """YOUR IMPLEMENTATION END"""

    return y


def conv2d_backward_x(w, x, input_x, strides=(1, 1), pad_size=(0, 0)):
    # Dilate and pad the input X before doing the convolution
    """YOUR IMPLEMENTATION START"""
    if strides[-1] > 1:
        x = dilate2d(x, strides)
    if pad_size[-1] > 0:
        x = pad2d(x, pad_size)
    """YOUR IMPLEMENTATION END"""

    bx, cx, hx, wx = x.shape
    bw, cw, hw, ww = w.shape

    y = np.zeros_like(input_x)
    by, cy, hy, wy = y.shape

    # Compute the convolution between X and W to get δEδx
    # Hint: use multiple for loops for the expected size
    """YOUR IMPLEMENTATION START"""
    for a in range(hw):
        for b in range(ww):
            x_slice = x[:, :, a:a + hy, b:b + wy]
            y[:, :, :, :] += np.einsum("lk,mlHW->mkHW", w[:, :, a, b], x_slice)
    """YOUR IMPLEMENTATION END"""

    return y


def conv2d_backward_w(w, x, input_w, strides=(1, 1), pad_size=(0, 0)):
    # Pad the input X and dilate the filter W before doing the convolution
    """YOUR IMPLEMENTATION START"""
    if pad_size[-1] > 0:
        x = pad2d(x, pad_size)
    if strides[-1] > 1:
        w = dilate2d(w, strides)
    """YOUR IMPLEMENTATION END"""

    bx, cx, hx, wx = x.shape
    bw, cw, hw, ww = w.shape

    y = np.zeros_like(input_w)
    by, cy, hy, wy = y.shape

    # Compute the convolution between X and W to get δEδw
    # Hint: use multiple for loops for the expected size
    """YOUR IMPLEMENTATION START"""
    # y shape is (bw, cw, hy, wy) which corresponds to (out_channels, in_channels, kh, kw)
    # w shape is (bx, bw, hw, ww) where hw, ww are output spatial dimensions
    # x shape is (bx, cx, hx, wx) where hx, wx are input spatial dimensions
    for i in range(hy):
        for j in range(wy):
            # w[:, :, a, b] for all a, b -> w[:, :, :, :]
            # x[:, :, i+a, j+b] for all a, b -> x[:, :, i:i+hw, j:j+ww]
            x_slice = x[:, :, i:i + hw, j:j + ww]
            # w is (bx, bw, hw, ww). x_slice is (bx, cx, hw, ww)
            # einsum("mk,ml->kl") where m is bx, k is bw, l is cx. But now we have spatial hw, ww.
            # We want to sum over bx (m) and spatial hw, ww (H, W).
            y[:, :, i, j] = np.einsum("mkHW,mlHW->kl", w, x_slice)
    """YOUR IMPLEMENTATION END"""

    return y


class Conv2d(ModelWithParameters):
    """
    A LinearRegression model applies a linear and bias function, in that order, to an input, ie
    y = wx+b, where w and b are the parameters of the Linear and Bias models,

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        kernel_initializer: Initializer = None,
        bias_initializer: Initializer = None,
        name=None,
    ):
        super().__init__(name=name)
        self.input_size = in_channels
        self.output_size = out_channels
        kh, kw = kernel_size
        stride_h, pad_size_h = stride, padding
        if kh == 1:  # Conv1d
            stride_h, pad_size_h = 1, 0
        self.strides = (stride_h, stride)
        self.pad_size = (pad_size_h, padding)
        if kernel_initializer is None:
            kernel_initializer = RandomNormal()
        shape = (out_channels, in_channels, kh, kw)
        w = kernel_initializer.create(shape)
        self.register_parameter("w", w)
        self.use_bias = bias
        if self.use_bias:
            if bias_initializer is None:
                bias_initializer = RandomNormal()
            b = bias_initializer.create((out_channels,))
            self.register_parameter("b", b)

    def forward(self, x: np.ndarray):
        y = {}

        # Retrieve w
        w = self.get_parameters()["w"]

        """YOUR IMPLEMENTATION START"""
        y = conv2d_forward(w, x, self.strides, self.pad_size)
        if self.use_bias:
            b_val = self.get_parameters()["b"]
            y = y + b_val[np.newaxis, :, np.newaxis, np.newaxis]
        """YOUR IMPLEMENTATION END"""

        # add input to cache to calculate δEδw in backward step
        self.set_cache(x)
        return y

    def backward(self, δEδy: np.ndarray):
        # Compute gradients for the parameters of the bias and convolution models
        δEδx, δEδw = {}, {}

        # Retrieve input from cache to calculate δEδw
        (x,) = self.get_cache()

        # Retrieve w
        w = self.get_parameters()["w"]

        """YOUR IMPLEMENTATION START"""
        ret_grads = {}
        if self.use_bias:
            δEδb = np.sum(δEδy, axis=(0, 2, 3))
            ret_grads["b"] = δEδb
            
        w_flipped = np.flip(w, axis=(2, 3))
        ph = w.shape[2] - 1 - self.pad_size[0]
        pw = w.shape[3] - 1 - self.pad_size[1]
        full_pad = (ph, pw)
        δEδx = conv2d_backward_x(w_flipped, δEδy, x, self.strides, full_pad)

        δEδw = conv2d_backward_w(δEδy, x, w, self.strides, self.pad_size)
        ret_grads["w"] = δEδw
        """YOUR IMPLEMENTATION END"""

        return δEδx, ret_grads
