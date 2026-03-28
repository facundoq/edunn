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
    import numpy as np
    import edunn as nn
    from edunn import utils

    return nn, np, utils


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Regression Model: `forward`

    A Linear Regression model is formed by first applying a linear layer $f(x)=wx$ and then a bias layer $f(x)=x+b$, resulting in the function $f(x)=wx+b$. Instead of viewing $f(x)$ as $wx+b$, we can see it as `x -> Linear -> Bias -> y`, i.e., as a sequence of layers, each transforming the input `x` to obtain the output `y`.

    In terms of code, the `forward` method of a `LinearRegression` model is the composition of the `forward` methods of the `Linear` and `Bias` layers:
    ```
    y_linear = linear.forward(x)
    y = bias.forward(y_linear)
    ```

    Implement the `forward` method of the `LinearRegression` model in the `edunn/models/linear_regression.py` file. Remember, that you don't need to perform any computation per se. Instead, chain the `forward` functions from other layers.
    """)
    return


@app.cell
def _(nn, np, utils):
    x = np.array([[3, -7], [-3, 7]])
    w = np.array([[2, 3, 4], [4, 5, 6]])
    b = np.array([1, 2, 3])
    linear_initializer = nn.initializers.Constant(w)
    bias_initializer = nn.initializers.Constant(b)
    _layer = nn.LinearRegression(2, 3, linear_initializer=linear_initializer, bias_initializer=bias_initializer)
    y = np.array([[-21, -24, -27], [23, 28, 33]])
    utils.check_same(y, _layer.forward(x))
    linear_initializer = nn.initializers.Constant(-w)
    bias_initializer = nn.initializers.Constant(-b)
    _layer = nn.LinearRegression(2, 3, linear_initializer=linear_initializer, bias_initializer=bias_initializer)
    utils.check_same(-y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Regression Model: `backward`

    The `backward` method of a `LinearRegression` model is the *inverse* composition of the `backward` methods of the `Linear` and `Bias` layers.

    That is, the derivative of the error at the output of the model first passes through `Bias`, which calculates the derivative with respect to its parameters and returns it as `δEδbias` (the only parameter is $b$ in this case). But `Bias` will also return `δEδx_bias`, the derivative of the error with respect its input. But the input to `Bias` is actuall the output of `Linear`. Therefore, we can do the same as with the `forward` to backpropagate the gradient, but in reverse order.

    This is the first (simple) example of the application of the *backpropagation* algorithm! This time, with only two models/layers. Later, we will generalize it with the `Sequential` model.

    In this case, we also help you by combining the gradient dictionaries of each layer into a single large gradient dictionary of `LinearRegression`. To achieve that, we use the `**` python operator, which unpacks a dictionary with `{**dict1, **dict2}`,  to combine them again afterwards.
    """)
    return


@app.cell
def _(nn, utils):
    samples = 100
    batch_size = 2
    din = 3  # input dimension
    dout = 5  # output dimension
    input_shape = (batch_size, din)
    _layer = nn.LinearRegression(din, dout)
    # Verify derivatives of a Linear Regression model
    # with random values for `w`, `b`, and `x`, the input
    utils.check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now have our first complex model! But we still can't train it. For that, we need:

    1. A dataset with samples (x,y)
    2. A loss function
    3. An optimization algorithm for that loss function that can run with the dataset and model.

    In the following guides, we will implement (2) and then (3). Using a sample dataset (1) from the `edunn` library we will test the Linear Regression model.
    """)
    return


if __name__ == "__main__":
    app.run()
