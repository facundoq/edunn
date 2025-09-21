import numpy as np
from ..model import ModelWithParameters

from ..initializers import Initializer, RandomNormal

from .bias import Bias


def dilate2d(x, dilation):
    b, c, h, w = x.shape
    dilation_h, dilation_w = dilation
    x_dilated = np.zeros((b, c, h + (h - 1) * (dilation_h - 1), w + (w - 1) * (dilation_w - 1)))

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
    for i in range(hy):
        for j in range(wy):
            for a in range(hw):
                for b in range(ww):
                    y[:, :, i, j] += np.einsum("mk,lk->lm", w[:, :, a, b], x[:, :, i * stride_h + a, j * stride_w + b])
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
    # Example: δEδx must be (2, 3, 7, 7) when X is (2,3,7,7) and W is (4,3,5,5)
    # Then w_flipped is (4, 3, 5, 5) and δEδy (2, 4, 3, 3)
    # When δEδy is padded (2, 4, 11, 11) which corresponds with int((hx - hd)) + 1 => 11-5+1=7
    for i in range(hy):
        for j in range(wy):
            for a in range(hw):
                for b in range(ww):
                    y[:, :, i, j] += np.einsum("lk,ml->mk", w[:, :, a, b], x[:, :, i + a, j + b])
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
    # Example: δEδw must be (4,3,5,5) when X is (2,3,7,7) and W is (4,3,5,5)
    # Then δEδy is (2,4,3,3) and X (2,3,7,7)
    for i in range(hy):
        for j in range(wy):
            for a in range(hw):
                for b in range(ww):
                    y[:, :, i, j] += np.einsum("mk,ml->kl", w[:, :, a, b], x[:, :, i + a, j + b])
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
        if kh == 1:  # Convolution1D
            stride_h, pad_size_h = 1, 0
        self.strides = (stride_h, stride)
        self.pad_size = (pad_size_h, padding)
        if kernel_initializer is None:
            kernel_initializer = RandomNormal()
        shape = (out_channels, in_channels, kh, kw)
        w = kernel_initializer.create(shape)
        self.register_parameter("w", w)
        # self.bias = Bias(out_channels, initializer=bias_initializer)

    def forward(self, x: np.ndarray):
        y = {}

        # Retrieve w
        w = self.get_parameters()["w"]

        """YOUR IMPLEMENTATION START"""
        y = conv2d_forward(w, x, self.strides, self.pad_size)
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
        w_flipped = np.flip(w, axis=(2, 3))
        ph = w.shape[2] - 1 - self.pad_size[0]
        pw = w.shape[3] - 1 - self.pad_size[1]
        full_pad = (ph, pw)
        δEδx = conv2d_backward_x(w_flipped, δEδy, x, self.strides, full_pad)

        δEδw = conv2d_backward_w(δEδy, x, w, self.strides, self.pad_size)
        """YOUR IMPLEMENTATION END"""

        return δEδx, {"w": δEδw}
