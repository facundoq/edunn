import numpy as np
from ..model import ModelWithParameters

from ..initializers import Initializer,RandomNormal

from .bias import Bias


def dilate2d(x, dilation):
    b, c, h, w = x.shape
    x_dilated = np.zeros((b, c, h + (h-1)*(dilation-1), w + (w-1)*(dilation-1)))

    ### YOUR IMPLEMENTATION START  ###
    x_dilated[:, :, ::dilation, ::dilation] = x
    ### YOUR IMPLEMENTATION END  ###

    return x_dilated

def pad2d(x, pad_size):
    b,c,h,w=x.shape
    new_shape = (b,c,h+2*pad_size,w+2*pad_size)
    x_padded = np.zeros(new_shape)

    ### YOUR IMPLEMENTATION START  ###
    x_padded[:,:,pad_size: -pad_size,pad_size: -pad_size] = x
    ### YOUR IMPLEMENTATION END  ###

    return x_padded

def is_odd(x):
    return x%2 == 1

def conv2d_forward(w, x, strides = (1,1), pad_size = 0):
    ## Pad the input X before doing the convolution
    ### YOUR IMPLEMENTATION START  ###
    if pad_size>0:
        x = pad2d(x, pad_size)
    ### YOUR IMPLEMENTATION END  ###

    stride_h,stride_w=strides

    bx, cx, hx, wx = x.shape
    bw, cw, hw, ww = w.shape
    assert is_odd(hw) and is_odd(ww), "The dimensions of w must be odd numbers"
    assert cx == cw, "The number of channels in the weight matrix must be equal to the number of channels in the image"

    hy = int((hx - hw) / stride_h) + 1
    wy = int((wx - ww) / stride_w) + 1
    y = np.zeros((bx, bw, hy, wy))

    # Compute the convolution between X and W to get Y
    # Hint: use multiple for loops for the expected size
    ### YOUR IMPLEMENTATION START  ###
    for i in range(hy):
        for j in range(wy):
            for a in range(hw):
                for b in range(ww):
                    for k in range(cx):
                        for l in range(bx):
                            for m in range(bw):
                                y[l,m,i,j] += w[m,k,a,b] * x[l,k,i*stride_h+a,j*stride_w+b]
    ### YOUR IMPLEMENTATION END  ###

    return y

def conv2d_backward_x(w, x, input_x, strides = (1,1), pad_size = 0):
    ## Dilate and pad the input X before doing the convolution
    ### YOUR IMPLEMENTATION START  ###
    if strides[0]>1:
        x = dilate2d(x, strides[0])
    if pad_size>0:
        x = pad2d(x, pad_size)
    ### YOUR IMPLEMENTATION END  ###

    bx,cx,hx,wx = x.shape
    bw,cw,hw,ww = w.shape

    y = np.zeros_like(input_x)
    by,cy,hy,wy = y.shape

    # Compute the convolution between X and W to get δEδx
    # Hint: use multiple for loops for the expected size
    ### YOUR IMPLEMENTATION START  ###
    # Example: δEδx must be (2, 3, 7, 7) when X is (2,3,7,7) and W is (4,3,5,5)
    # Then w_flipped is (4, 3, 5, 5) and δEδy (2, 4, 3, 3)
    # When δEδy is padded (2, 4, 11, 11) which corresponds with int((hx - hd)) + 1 => 11-5+1=7
    for i in range(hy):
        for j in range(wy):
            for a in range(hw):
                for b in range(ww):
                    for k in range(cw):
                        for l in range(bw):
                            for m in range(by):
                                y[m,k,i,j] += w[l,k,a,b] * x[m,l,i+a,j+b]
    ### YOUR IMPLEMENTATION END  ###

    return y

def conv2d_backward_w(w, x, input_w, strides = (1,1), pad_size = 0):
    ## Pad the input X and dilate the filter W before doing the convolution
    ### YOUR IMPLEMENTATION START  ###
    if pad_size>0:
        x = pad2d(x, pad_size)
    if strides[0]>1:
        w = dilate2d(w, strides[0])
    ### YOUR IMPLEMENTATION END  ###

    bx,cx,hx,wx = x.shape
    bw,cw,hw,ww = w.shape

    y = np.zeros_like(input_w)
    by,cy,hy,wy = y.shape

    # Compute the convolution between X and W to get δEδw
    # Hint: use multiple for loops for the expected size
    ### YOUR IMPLEMENTATION START  ###
    # Example: δEδw must be (4,3,5,5) when X is (2,3,7,7) and W is (4,3,5,5)
    # Then δEδy is (2,4,3,3) and X (2,3,7,7)
    for i in range(hy):
        for j in range(wy):
            for a in range(hw):
                for b in range(ww):
                    for k in range(by):
                        for l in range(cy):
                            for m in range(bx):
                                y[k,l,i,j] += w[m,k,a,b] * x[m,l,i+a,j+b]
    ### YOUR IMPLEMENTATION END  ###

    return y

class Convolution2D(ModelWithParameters):
    '''
    A LinearRegression model applies a linear and bias function, in that order, to an input, ie
    y = wx+b, where w and b are the parameters of the Linear and Bias models,

    '''

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tuple, stride: int = 1, padding: int = 0, bias: bool = True,
                 kernel_initializer: Initializer = None, bias_initializer: Initializer = None, name=None):
        super().__init__(name=name)
        self.input_size=in_channels
        self.output_size=out_channels
        self.strides = (stride, stride)
        self.pad_size = padding
        if kernel_initializer is None:
            kernel_initializer = RandomNormal()
        kh,kw=kernel_size
        shape = (out_channels,in_channels,kh,kw)
        w = kernel_initializer.create(shape)
        self.register_parameter("w", w)
        # self.bias = Bias(out_channels, initializer=bias_initializer)

    def forward(self, x: np.ndarray):
        y = {}

        # Retrieve w
        w = self.get_parameters()["w"]

        ### YOUR IMPLEMENTATION START  ###
        y = conv2d_forward(w,x,self.strides,self.pad_size)
        ### YOUR IMPLEMENTATION END  ###

        # add input to cache to calculate δEδw in backward step
        self.set_cache(x)
        return y

    def backward(self, δEδy: np.ndarray):
        # Compute gradients for the parameters of the bias and convolution models
        δEδx, δEδw = {}, {}

        # Retrieve input from cache to calculate δEδw
        x, = self.get_cache()

        # Retrieve w
        w = self.get_parameters()["w"]

        ### YOUR IMPLEMENTATION START  ###
        w_flipped = np.flip(w,axis=(2,3))
        full_pad = w.shape[-1]-1-self.pad_size
        δEδx = conv2d_backward_x(w_flipped,δEδy,x,self.strides,full_pad)

        δEδw = conv2d_backward_w(δEδy,x,w,self.strides,self.pad_size)
        ### YOUR IMPLEMENTATION END  ###

        return δEδx, {"w":δEδw}
