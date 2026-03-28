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

    import edunn as nn
    import numpy as np

    return nn, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dense Layer

    The `Linear`, `Bias`, and activation layers (`Sigmoid`, `ReLU`, `TanH`, etc.) are often used together in the form `dense(x) = activation(w*x+b)`, where `activation` is an activation function. This layer is commonly referred to as `FullyConnected` or, as we'll call it here, `Dense`, and the name comes from the fact that each output of the layer depends on *all* inputs, plus a few bells and whistles.

    In this exercise, you need to implement the `Dense` layer. But don't do it from scratch;use the `Linear`, `Bias`, and activation layers directly without copying their code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creation and Initialization

    The `Dense` layer should have a parameter vector `w`, another vector `b`, and a specific activation function.

    To implement it, we will use three *internal layers*: `Linear`, `Bias`, and the activation layer, which we'll call `Activation`. For convenience, we'll also allow specifying the activation using a string like `relu`, `sigmoid`, or `tanh`. In this case, we've already defined the constructor `__init__`, which assigns the corresponding internal layer objects `self.linear`, `self.bias`, and `self.activation`, and allows specifying the initializers for each of them.

    We've also provided you with the implementation of `get_parameters`, which combines the parameter dictionaries of each sub-layer into a single dictionary of gradients for the `Dense` layer.

    We recommend studying the code of these two methods (`__init__` and `get_parameters`) to understand how they work. They will help you implement the `forward` and `backward` methods for `Dense`.
    """)
    return


@app.cell
def _(nn):
    # Create a Dense layer with 2 input values and 3 output values
    # and ReLU activation
    # The linear layer is initialized randomly
    # While the bias layer is initialized with zeros

    input_dimension=2
    output_dimension=3
    activation="relu"
    dense1=nn.Dense(input_dimension,output_dimension,
                     activation_name="relu",
                     linear_initializer=nn.initializers.RandomNormal(),
                     bias_initializer=nn.initializers.Constant(0),
                     )
    print(f"Layer name: {dense1.name}")
    print(f"Layer's parameter w: {dense1.get_parameters()['w']}")
    print("(should change every time you run this cell)")
    print(f"Layer's parameter b: {dense1.get_parameters()['b']}")
    print("(should always be 0)")
    print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forward Method

    Now that we know how to create and initialize `Dense` layer objects, let's start with the `forward` method, which you can find in the `dense.py` file in the `edunn/models` folder.

    To implement the forward pass, you should take the input `x` and use it to call the `forward` method of the internal layers of type `Linear`, `Bias`, and `Activation`.

    To verify that the `forward` implementation is correct, we use the `Constant` initializer twice, but afterward, the layer continues to use a random initializer like `RandomNormal` by default.
    """)
    return


@app.cell
def _(nn, np):
    x = np.array([[3, -7], [-3, 7]])
    w = np.array([[2, 3, 4], [4, 5, 6]])
    b = np.array([1, 2, 3])
    linear_initializer = nn.initializers.Constant(w)
    bias_initializer = nn.initializers.Constant(b)
    _layer = nn.Dense(2, 3, linear_initializer=linear_initializer, bias_initializer=bias_initializer)
    y = np.array([[-21, -24, -27], [23, 28, 33]])
    nn.utils.check_same(y, _layer.forward(x))
    linear_initializer = nn.initializers.Constant(-w)
    bias_initializer = nn.initializers.Constant(-b)
    _layer = nn.Dense(2, 3, linear_initializer=linear_initializer, bias_initializer=bias_initializer)
    nn.utils.check_same(-y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backward Method

    To implement the `backward` method, you should also call the `backward` method of the `self.linear`, `self.bias`, and `self.activation` variables in the correct order and manner. Hint: it's the reverse of the `forward` method.

    In this case, we also help you by combining the gradient dictionaries of each layer into a single large gradient dictionary for `Dense` using the `**dict` operator, which unpacks a dictionary, and `{**dict1, **dict2}`, which combines them again.
    """)
    return


@app.cell
def _(nn):
    samples = 100
    batch_size = 2
    features_in = 3
    features_out = 5
    input_shape = (batch_size, features_in)
    _layer = nn.Dense(features_in, features_out, activation_name='relu')
    # Test derivatives of a Dense layer with random values for `w`
    nn.utils.check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


if __name__ == "__main__":
    app.run()
