import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo

    from edunn import utils
    import edunn as nn
    import numpy as np

    return nn, np, utils


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dense or Fully Connected or Linear Regression Layers

    The most common layer in a neural network is a layer that implements the function `y = xw+b`, where `x` is the input, `y` is the output, and `b` is a bias vector, and `w` is a weight matrix. However, implementing this layer can be challenging. Instead, we will separate the implementation into two parts.

    * The `Bias` layer, which only adds `b` to its input, i.e., `y = x + b`
    * The `Linear` layer, which only multiplies its input by the weight matrix `w`, i.e., `y = w * x`
    * By combining these two layers, we can achieve the functionality of the traditional layer called `Dense` or `FullyConnected` in other libraries. This will allows us to use  a linear regression model with the function `y(x) = wx + b` to solve many problems!

    We will begin with the `Bias` layer, the simpler of the two.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bias Layer

    In this exercise, you need to implement the `Bias` layer, which adds a different value to each of its inputs to generate its output. This value is _not_ constant but rather a parameter of the network.

    For example, if the input `x` is `[3.5, -7.2]` and the `Bias` layer has parameters `[2.0, 3.0]`, then the output `y` will be `[5.5, -4.2]`.

    Your goal is to implement the `forward` and `backward` methods for this layer so that it can be used in a neural network.

    This layer works for arrays that have the same size as the `Bias` layer's parameters (excluding the batch dimension).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creation and Initialization

    The `Bias` layer has a parameter vector `b`, which must be created and initialized in some way. Additionally, this parameter is registered in the layer so that it can be accessed later.

    Take a look at the file `edunn/models/bias.py` for the implementation of the `__init__` method of the `Bias` layer to see how the weight is created. You'll see that it will employ an `Initializer` to determine its initial value.

    When creating the layer, we can pass an object of type `Initializer` that will create and assign the initial value to the parameter `b`. By default, `b` is initialized with zeros using the `initializers.Zero` class.

    Take a look at the implementation of the `Zero` class in `edunn/initializers.py`. Examining the implementation of this class, we can see that:
    * It inherits from `Initializer`
    * It implements the `initialize(self, p: np.ndarray)` method, which receives a numpy array for initialization
    * It uses `p[:]` to initialize to 0 instead of `p = 0`. There are two important reasons for this:
        * Using `p = 0` would only change the local variable `p` instead of changing the numpy array that `p` points to.
        * By using `p[:]`, we are changing the __content__ of the parameter array, which belongs to a class like `Bias` or another that we will implement later.

    Once the class is created, we can obtain the parameter vector `p` from the `Bias` class using the `get_parameters()` method.
    """)
    return


@app.cell
def _(nn):
    # Create a Bias layer with 2 input/output values
    _bias = nn.Bias(2, initializer=nn.initializers.Zero())
    print(f'Layer Name: {_bias.name}')
    print(f'Layer Parameters: {_bias.get_parameters()}')
    print()
    _bias2 = nn.Bias(2)
    # By default, the initializer is 'Zero'
    print(f'Layer Name: {_bias2.name}')
    print(f'Layer Parameters: {_bias.get_parameters()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Accessing Parameters by Name

    The `get_parameters()` method returns a dictionary of parameters, because a layer can have more than one parameter. Given that we already know the name of the only parameter in this layer, we can access it by its name in string form, `'b'`:
    """)
    return


@app.cell
def _(nn):
    # Create a Bias layer with 2 input/output values
    _bias = nn.Bias(2, initializer=nn.initializers.Zero())
    print(f'Layer Name: {_bias.name}')
    print(f"Layer Parameter 'b': {_bias.get_parameters()['b']}")
    print()
    _bias2 = nn.Bias(2)
    # By default, the initializer is 'Zero'
    print(f'Layer Name: {_bias2.name}')
    print(f"Layer Parameter 'b': {_bias2.get_parameters()['b']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Implementation of a Constant Initializer

    While sometimes parameters are initialized to '0', it's most common to initialize them with some constant value.

    Before starting with the implementation of the `Bias` class, you should implement the `Constant` initializer that assigns a constant value or array to the parameter. This allows, for example, initializing `b` with all values of `3` or with a vector of values `[1, 2, 3, 4]`.

    Find the `Constant` class in the `edunn/initializers.py` module and implement the `initialize` method.
    """)
    return


@app.cell
def _(nn, np, utils):
    # Create a Bias layer with 2 output values (and input values as well)
    # All parameters are initialized to 3
    value = 3
    _bias = nn.Bias(2, initializer=nn.initializers.Constant(value))
    print(f'Layer Name: {_bias.name}')
    print(f"Layer Parameter 'b': {_bias.get_parameters()['b']}")
    utils.check_same(_bias.get_parameters()['b'], np.array([3, 3]))
    print()
    value = np.array([1, 2, 3, 4])
    _bias = nn.Bias(4, initializer=nn.initializers.Constant(value))
    # Create a Bias layer with initial values 1, 2, 3, 4. Note that we are ensuring that the number of values of the Constant initializer match those of the bias array
    print(f'Layer Name: {_bias.name}')
    print(f"Layer Parameter 'b': {_bias.get_parameters()['b']}")
    utils.check_same(_bias.get_parameters()['b'], value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forward Method

    Now that we know how to create and initialize `Bias` layer objects, let's begin with the `forward` method, which can be found in the file `edunn/models/bias.py`.

    If the parameters to be added are $[b_1, ..., b_f]$ and the input to the layer is $x = [x_1, x_2, ..., x_f]$, then the output $y$ is:

    $
    y([x_1, x_2, ..., x_f]) = [x_1 + b_1, x_2 + b_2, ..., x_f + b_f]
    $
    """)
    return


@app.cell
def _(nn, np, utils):
    x = np.array([[3.5, -7.2, 5.3], [-3.5, 7.2, -5.3]])
    _initializer = nn.initializers.Constant(np.array([2, 3, 4]))
    _layer = nn.Bias(3, initializer=_initializer)
    y = np.array([[5.5, -4.2, 9.3], [-1.5, 10.2, -1.3]])
    utils.check_same(y, _layer.forward(x))
    _initializer = nn.initializers.Constant(-np.array([2, 3, 4]))
    _layer = nn.Bias(3, initializer=_initializer)
    y = np.array([[1.5, -10.2, 1.3], [-5.5, 4.2, -9.3]])
    utils.check_same(y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backward Method

    In addition to calculating its output, the
    layer must also be able to backpropagate the gradient of the network's error. Therefore, you need to implement the `backward` method, which receives $\frac{δE}{δy}$, the partial derivatives of the error with respect to the output (gradient) of this layer, and returns $\frac{δE}{δx}$, the partial derivatives of the error with respect to the inputs of this layer.

    ## `δEδx`
    For the `Bias` layer, the gradient calculation with respect to the input `δEδx` is simple since it is the same as the case with the `AddConstant` layer.

    $ \frac{δE}{δx} =\frac{δE}{δy} $

    $
    y([x_1, x_2, ..., x_f]) = [x_1 + b_1, x_2 + b_2, ..., x_n + b_f]
    $

    Then, using the chain rule:

    $
    \frac{δE}{δb_i} = \frac{δE}{δy} \frac{δy_i}{δb_i} = \frac{δE}{δy_i} \frac{δy_i}{δb_i} = \frac{δE}{δy_i} \frac{δ (x_i + b_i)}{δb_i} = \frac{δE}{δy_i} 1 = \frac{δE}{δy_i}
    $

    In other words,

    $ \frac{δE}{δx} =\frac{δE}{δy} $

    ## `δEδb`

    For this layer, you also need to calculate the gradient with respect to the parameters `b` so that they can be optimized to minimize the error. Therefore, you also need to calculate `δEδb`. Remember that:

    However, in the case of the gradient of the error with respect to `b`, the formula is the same, $ \frac{δE}{δb} =\frac{δE}{δy}$. This is because $\frac{δy_i}{δb_i} = \frac{δ(x_i + b_i)}{δb_i} = \frac{δ(x_i + b_i)}{δx_i} =1$. In other words, if we view both `b` and `x` as inputs to the layer, $x + b$ is symmetric in `x` and `b`, and thus, their derivatives are also symmetric.
    """)
    return


@app.cell
def _(nn, np, utils):
    # Number of random values and batch size
    # to generate values of x and δEδy for gradient checking
    samples = 100
    batch_size = 2
    features = 4
    # Dimensions of input and output for the layer, and initializer
    input_shape = (batch_size, features)
    _initializer = nn.initializers.Constant(np.array(range(features)))
    _layer = nn.Bias(features)
    utils.check_gradient.common_layer(_layer, input_shape, samples=samples)
    # Verify gradients of a Bias layer with b=[0,1,2,3]
    _initializer = nn.initializers.Constant(-np.array(range(features)))
    _layer = nn.Bias(features)
    # Verify gradients of a Bias layer with b=[0,-1,-2,-3]
    utils.check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


if __name__ == "__main__":
    app.run()
