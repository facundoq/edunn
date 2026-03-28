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
    # Softmax Layer

    For classification problems, it's useful to have a model that can generate probability distributions as output. In this way, for a problem with `C` classes, the model can output a vector of `C` elements `f(x) = y`, where each `y_i` is a value between 0 and 1, indicating the probability that example `x` belongs to class `i`. Furthermore, since `y` represents a distribution, it must sum to one. Formally:

    $$\sum_{i=1}^C y_i = 1$$

    A linear regression model can generate a vector of `C` elements with the _scores_ for each class, but these values will be in the range $(-\infty, +\infty)$, and thus, they cannot satisfy the properties of a probability distribution mentioned above. However, we can turn those scores into a probability distribution with a $softmax$ function.

    In this exercise, you need to implement the `Softmax` layer, which, given a vector `x` of `C` scores per class, converts it into a vector `y` of probabilities per class. To do this, implement the Softmax function. Formally:

    $$y =(y_1,y_2,...,y_C) = Softmax((x_1,x_2,...,x_C)) = Softmax(x)$$

    Where each $y_i$ is the probability of class $i$.

    For example, given class scores `[-5,100,100]`, the `Softmax` function will generate probabilities `[0,0.5,0.5]`.

     We call layers such as `Softmax` **activations** because they have no parameters and simply modify the values of a vector/tensor, generally in some non-linear way. You have actually encountered activation functions before, in the `AddConstant` and `MultiplyConstant` layers!

    Implemment `Softmax`, which can be found in `edunn/models/activations.py`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forward Method

    The `forward` method uses the following formula for `y`:

    $$y=
    \frac{[e^{x_1},...,e^{x_c}]}{e^{x_1}+...+e^{x_c}} =
    \frac{[e^{x_1},...,e^{x_c}]}{N}$$

    Or, viewed element by element, each value of $y$ is defined as:
    $$y_i(x) =  \frac{e^{x_i}}{e^{x_1}+...+e^{x_c}} $$

    Here, we use the exponential function ($e^x$) to transform each score in the `x` vector from the range $(-\infty, +\infty)$ to the range $(0, +\infty)$ since the exponential function can only output zero or positive values.

    Furthermore, $e^x$ is monotonically increasing in $x$, so higher values of $x$ lead to higher values of $e^x$, meaning that if a score is high, the probability will also be high, and vice versa.

    Now, in addition, each element is divided by the value $N$, which is used to normalize the values, achieving:
    1. Values between 0 and 1
    2. The sum of values equals 1

    That is, the axioms of a probability distribution as mentioned earlier.

    Implement the `forward` method of the `Softmax` class.
    """)
    return


@app.cell
def _(nn, np, utils):
    x = np.array([[0, 0, 100], [0, 100, 0.0], [100, 100, 0.0], [50, 50, 0.0], [1, 1, 1]], dtype=float)
    _layer = nn.Softmax()
    y = np.array([[0, 0, 1], [0, 1, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]], dtype=float)
    utils.check_same(y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backward Method

    The `backward` method of the Softmax function requires several steps because, due to normalization, each output of Softmax depends on each input.

    To keep this notebook concise, the details of the derivative calculation can be found in [this online resource](http://facundoq.github.io/edunn/material/en/softmax.html).

    Implement the `backward` method of the `Softmax` class:
    """)
    return


@app.cell
def _(nn):
    from edunn.utils import check_gradient
    samples = 100
    input_shape = (5, 2)
    # number of random values of x and δEδy to generate and test gradients
    _layer = nn.Softmax()
    # Test derivatives of an AddConstant layer that adds 3
    check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


if __name__ == "__main__":
    app.run()
