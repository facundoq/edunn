import numpy as np
import scipy.special
import scipy.signal
import scipy.ndimage

def flatten_forward(x):
    return x.reshape(x.shape[0], -1)

def flatten_backward(g, x_shape):
    return g.reshape(x_shape)

def gelu_forward(x):
    return 0.5 * x * (1 + scipy.special.erf(x / np.sqrt(2)))

def gelu_backward(g, x):
    return g * (0.5 * (1 + scipy.special.erf(x / np.sqrt(2))) + np.exp(-0.5*x**2)*x / np.sqrt(2*np.pi))

def dropout_forward(x, p=0.5):
    mask = np.random.binomial(1, p, size=x.shape)
    y_ref = (x * mask) / p
    return y_ref, mask

def dropout_backward(g, mask, p=0.5):
    return (g * mask) / p

def batchnorm_forward(x, w, b, eps=1e-5):
    mu = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mu) / np.sqrt(var + eps)
    return w * x_norm + b

def maxpool2d_forward(x, kernel_size, stride):
    # Reference implementation assuming stride == kernel_size
    assert kernel_size == stride, "Reference maxpool assumes stride == kernel_size"
    return x.reshape(x.shape[0], x.shape[1], x.shape[2]//stride, stride, x.shape[3]//stride, stride).max(axis=(3, 5))

def avgpool2d_forward(x, kernel_size, stride):
    # Reference implementation using scipy
    y_ref_avg = scipy.ndimage.uniform_filter(x, size=(1, 1, kernel_size, kernel_size), mode="constant", cval=0.0)
    return y_ref_avg[:, :, ::stride, ::stride]

def conv2d_forward(x, w, stride=1, padding=0):
    # Reference implementation for 2D cross-correlation (convolution)
    if padding > 0:
        x = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)))
    y_ref = np.zeros((x.shape[0], w.shape[0], int((x.shape[2] - w.shape[2]) / stride + 1), int((x.shape[3] - w.shape[3]) / stride + 1)))
    for b in range(x.shape[0]):
        for f in range(w.shape[0]):
            for c in range(x.shape[1]):
                y_ref[b, f] += scipy.signal.correlate2d(x[b, c], w[f, c], mode="valid")[::stride, ::stride]
    return y_ref
