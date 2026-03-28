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
    # Squared Error Layer

    In this exercise, you need to implement the error layer `SquaredError` found in `eduu/models/squared_error.py`, which allows you to calculate the error for a batch of examples.

    Error layers are different from normal layers for two reasons:

    1. They not only take the output of the previous layer as input but also the expected value from the previous layer (`y` and `y_true`).
    2. For a batch of $n$ examples, their output is a vector of size $n$. In other words, they indicate the error value for each example with a scalar (real number).

    The error layer should also be able to perform the `backward` operation to propagate the error gradient backward through the network.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forward Method

    The `forward` method of the `SquaredError` layer should simply calculate the squared Euclidean distance between `y`, the values produced by the network, and `y_true`, the expected values.

    For example, if $y=[2,-2]$ and $y_{true}=[3,3]$, then the output of the layer is:

    $$E(y,y_{true})=d_2(y,y_{true})=d_2([2,-2],[3,3])=(2-3)^2+(-2-3)^2 = 1^2+(-5)^2=26$$

    In general, given two vectors $a=[a_1,\dots,a_n]$ and $b=[b_1,\dots,b_n]$, the squared Euclidean distance $d_2$ is:

    $$
    d_2(a,b) = d_2([a_1,\dots,a_n],[b_1,\dots,b_n]) =(a_1-b_1)^2+\dots+(a_n-b_n)^2
    $$

    In the case of a batch of examples, the calculation is independent for each example. It's important that the sum of squared differences is calculated per example (row) and not per feature (column).
    """)
    return


@app.cell
def _(nn, np, utils):
    y = np.array([[2, -2], [-4, 4]])
    y_true = np.array([[3, 3], [-5, 2]])
    _layer = nn.SquaredError()
    E = np.array([[26], [5]])
    utils.check_same(E, _layer.forward(y, y_true))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backward Method

    Now that you can calculate the error of a network, great! This is the final layer of the network when it's being trained. Therefore, the backward method of an error layer does not receive $\frac{δE}{δy}$; in fact, it should calculate it directly from $y$, $y_{true}$, and the error definition. Also, there are no parameters involved.

    So, in this case, the derivative is simple. We just need to calculate $\frac{δE}{δy}$, the derivative of the error with respect to the output computed by the network, $y$.
    In this case, $E$ is symmetric with respect to its inputs, so let's call it $a$ and $b$ again, and then calculate the derivative with respect to element $i$ of $a$ (the derivative with respect to $b$ would be the same):

    $$
    \frac{δE(a,b)}{δa_i} = \frac{δ((a_1-b_1)^2+\dots+(a_n-b_n)^2)}{δa_i} \\
    = \frac{δ((a_i-b_i)^2)}{δa_i} = 2 (a_i-b_i) \frac{δ((a_i-b_i))}{δa_i} \\
    = 2 (a_i-b_i) 1 = 2 (a_i-b_i)
    $$
    Generalizing for the entire vector $a$, we get:
    $$
    \frac{δE(a,b)}{δa} = 2 (a-b)
    $$
    Where $a-b$ is a vector subtraction.

    Again, as this error is calculated for each example, the calculations are independent for each row.
    """)
    return


@app.cell
def _(nn, utils):
    # number of random values of x and δEδy to generate and test gradients
    samples = 100
    batch_size = 2
    features_in = 3
    features_out = 5
    input_shape = (batch_size, features_in)
    _layer = nn.SquaredError()
    utils.check_gradient.squared_error(_layer, input_shape, samples=samples)
    return


if __name__ == "__main__":
    app.run()
