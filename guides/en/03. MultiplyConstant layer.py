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
    # MultiplyConstant Layer

    In this exercise, you need to implement the `MultiplyConstant` layer, which multiplies each of its inputs by a constant value to generate its output. It works similarly to `AddConstant`, but in this case, it performs multiplication instead of addition, and thus its derivatives are slightly more complicated.

    For example, if the input `x` is `[3.5, -7.2, 5.3]` and the `MultiplyConstant` layer is created with the constant `2`, then the output `y` will be `[7.0, -14.4, 10.6]`.

    Your goal is to implement the `forward` and `backward` methods for this layer so that it can be used in a neural network.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forward Method

    The `forward` method calculates the output `y` based on the input `x`, as explained above. In formal terms, if the constant to be multiplied is $C$ and the input to the layer is $x = [x_1, x_2, ..., x_n]$, then the output $y$ is:

    $
    y([x_1, x_2, ..., x_n]) = [x_1 * C, x_2 * C, ..., x_n * C]
    $

    We start with the `forward` method of the `MultiplyConstant` class, which can be found in the `activations.py` file in the `edunn/models` folder. You need to complete the code between the comments:

    ```
    \"\"\"YOUR IMPLEMENTATION START\"\"\"
    ```
    and

    ```
    \"\"\"YOUR IMPLEMENTATION END\"\"\"

    ```

    Then, verify your implementation with the following cell that checks with a layer that multiplies by 2 and another that multiplies by -2. If both checks are correct, you will see two messages with <span style='background-color:green;color:white;'>success</span>.
    """)
    return


@app.cell
def _(nn, np, utils):
    x = np.array([[3.5, -7.2, 5.3], [-3.5, 7.2, -5.3]])
    _layer = nn.MultiplyConstant(2)
    y = np.array([[7.0, -14.4, 10.6], [-7.0, 14.4, -10.6]])
    utils.check_same(y, _layer.forward(x))
    _layer = nn.MultiplyConstant(-2)
    y = -np.array([[7.0, -14.4, 10.6], [-7.0, 14.4, -10.6]])
    utils.check_same(y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backward Method

    In addition to calculating the output of the layer, it must be able to propagate the gradient of the network's error backward. To do this, you need to implement the `backward` method, which receives $\frac{δE}{δy}$, the partial derivatives of the error with respect to the output (gradient) of this layer, and returns $\frac{δE}{δx}$, the partial derivatives of the error with respect to the inputs of this layer.

    For the `MultiplyConstant` layer, calculating the gradient is slightly more complicated than for `AddConstant`. Let's go through the derivation:

    We have:

    $y_i(x) = x_i * C$

    And we want to calculate $\frac{δE}{δx_i}$. Using the chain rule, we can write:

    $\frac{δE}{δx_i} = \frac{δE}{δy} * \frac{δy}{δx_i}$

    Now, $\frac{δy}{δx_i}$ is straightforward as it's just the constant $C$:

    $\frac{δy}{δx_i} = C$

    So, we have:

    $\frac{δE}{δx_i} = \frac{δE}{δy} * C$

    In vector form for the input vector $x$, we get:

    $\frac{δE}{δx} = [\frac{δE}{δy_1} * C, \frac{δE}{δy_2} * C, ..., \frac{δE}{δy_n} * C] = \frac{δE}{δy} * C$

    So, the layer simply propagates the gradients from the next layer, but multiplied by the constant $C$.

    Complete the code in the `backward` function of the `MultiplyConstant` layer between the comments:

    ```
    \"\"\"YOUR IMPLEMENTATION START\"\"\"
    ```
    and

    ```
    \"\"\"YOUR IMPLEMENTATION END\"\"\"

    ```

    Then, verify with the following cell for a layer that multiplies by 3 and another that multiplies by -4. If both checks are correct, you will see two messages with <span style='background-color:green;color:white;'>success</span>.
    """)
    return


@app.cell
def _(nn):
    from edunn.utils import check_gradient
    samples = 100
    # Number of random values of x and δEδy to generate and test gradients
    input_shape = (5, 2)
    _layer = nn.MultiplyConstant(3)
    check_gradient.common_layer(_layer, input_shape, samples=samples)
    _layer = nn.MultiplyConstant(-4)
    # Test derivatives of a MultiplyConstant layer that multiplies by 3
    # Test derivatives of a MultiplyConstant layer that multiplies by -4
    check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


if __name__ == "__main__":
    app.run()
