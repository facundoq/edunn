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
    # AddConstant Layer

    To start easy, in this exercise you'll need to implement the `AddConstant` layer, which adds a constant value to each of its inputs to generate its output.

    For example, if the input `x` is `[3.5, -7.2, 5.3]` and the AddConstant layer adds the value `3.0` to its input, then the output `y` will be `[6.5, -4.2, 8.3]`.

    Your goal is to implement the `forward` and `backward` methods for this layer so that it can be used in a neural network.

    This layer works for input arrays of any size, including vectors, matrices, or arrays with more dimensions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forward Method

    The `forward` method calculates the output `y` based on the input `x`, as explained above. In formal terms, if the constant to be added is $C$ and the input to the layer is $x = [x_1, x_2, ..., x_n]$, then the output $y$ is:

    $
    y([x_1, x_2, ..., x_n]) = [x_1 + C, x_2 + C, ..., x_n + C]
    $

    We start with the `forward` method of the `AddConstant` class, which can be found in the `activations.py` file in the `edunn/models` folder. You need to complete the code between the comments:

    ```
    \"\"\"YOUR IMPLEMENTATION START\"\"\"
    ```
    and

    ```
    \"\"\"YOUR IMPLEMENTATION END\"\"\"
    ```

    Then, verify with the following cell for a layer that adds 3 and another that adds -3. If both checks are correct, you will see two messages with <span style='background-color:green;color:white;'>success</span>.
    """)
    return


@app.cell
def _(nn, np, utils):
    x = np.array([[3.5, -7.2, 5.3], [-3.5, 7.2, -5.3]])
    _layer = nn.AddConstant(3)
    y = np.array([[6.5, -4.2, 8.3], [-0.5, 10.2, -2.3]])
    utils.check_same(y, _layer.forward(x))
    _layer = nn.AddConstant(-3)
    y = np.array([[0.5, -10.2, 2.3], [-6.5, 4.2, -8.3]])
    utils.check_same(y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backward Method

    In addition to calculating its output with `forward`, the layer must also be able to propagate the gradient of the network's error backward. To do this, you need to implement the `backward` method, which receives $\frac{δE}{δy}$, the partial derivatives (gradient) of the error with respect to the output of this layer, and returns $\frac{δE}{δx}$, the partial derivatives of the error with respect to the inputs of this layer.

    For the `AddConstant` layer, calculating the gradient of the output $y$ with respect to the input $x$ is straightforward. However, we need to return the gradient of $E$ with respect to $x$. Let's recall the form of the output:

    $
    y([x_1, x_2, ..., x_n]) = [x_1 + C, x_2 + C, ..., x_n + C]
    $

    Our goal is to calculate $\frac{δE}{δx_i}$. For that, we'll focus on $\frac{δE}{δx_i}$, which is the derivative of the error with respect to a specific input. Then, applying the chain rule, we can write $\frac{δE}{δx_i}$ as:

    $\frac{δE}{δx_i} = \frac{δE}{δy} * \frac{δy}{δx_i} = \sum_j \frac{δE}{δy_j} * \frac{δy_j}{δx_i}$

    We can consider each element of the output separately:

    $y_i(x) = x_i + C$

    And therefore:

    $\frac{δy_i}{δx_i} = 1 + 0 = 1$

    Since there is no interaction between elements with different indices (i.e., $y_i$ depends only on $x_i$), in the sum above, if $i \neq j$, then $\frac{δy_j}{δx_i} = 0$. Thus, we can remove the summation and use the chain rule only with $y_i$:

    $\frac{δE}{δx_i} = \frac{δE}{δy} * \frac{δy}{δx_i} = \sum_j \frac{δE}{δy_j} * \frac{δy_j}{δx_i} = \frac{δE}{δy_i} * \frac{δy_i}{δx_i}$

    Knowing that $\frac{δy_i}{δx_i} = 1$

    $\frac{δE}{δx_i} = \frac{δE}{δy} * \frac{δy}{δx_i} = \frac{δE}{δy_i} * 1 = \frac{δE}{δy_i}$

    Writing it in vector form for the vector x:

    $\frac{δE}{δx} = [\frac{δE}{δy_1}, \frac{δE}{δy_2}, ..., \frac{δE}{δy_n}] = \frac{δE}{δy}$

    What dos this equation mean? Simply that the layer just propagates the gradients from the next layer to the previous one.

    Note that for simplicity, in the code we call these vectors `δEδy` and `δEδx`. Also, remember that in this case $C$ is a constant and NOT a network parameter, so we do not need to calculate $\frac{δE}{δC}$.

    How do we know if the implementation is correct? Well, gradient checking is done with the `check_gradient_layer_random_sample` function. This function generates random samples of `x` and `δEδy`, and then compares the analytical gradient (your implementation) with the *numerical gradient*. The numerical gradient is costly to compute, but can be computed with the same algorithm for different functions.

    The numerical gradient _approximates_ the partial derivatives using the derivative formula:

    $\frac{δf(x)}{δx} = \lim_{h→0} \frac{f(x+h) - f(x)}{h}$

    with a very small value of $h$ ($h=10^{-12}$). In truth, for a better approximation, it uses the _centered derivative_, whose formula is:

     $\frac{δf(x)}{δx} \~= \frac{f(x+h) - f(x-h)}{2h}$.

    This gradient checking technique is a standard method to verify the correct implementation of a neural network.

    Complete the code in the `backward` function of the `AddConstant` layer between the comments:

    ```
    \"\"\"YOUR IMPLEMENTATION START\"\"\"
    ```
    and

    ```
    \"\"\"YOUR IMPLEMENTATION END\"\"\"
    ```

    Then, verify with the following cell for a layer that adds 3 and another that adds -3. If both checks are correct, you will see two messages with <span style='background-color:green;color:white;'>success</span>.
    """)
    return


@app.cell
def _(nn):
    from edunn.utils import check_gradient
    samples = 100
    # Number of random values of x and δEδy to generate and test gradients
    input_shape = (5, 2)
    _layer = nn.AddConstant(3)
    check_gradient.common_layer(_layer, input_shape, samples=samples)
    _layer = nn.AddConstant(-4)
    # Test derivatives of an AddConstant layer that adds 3
    # Test derivatives of an AddConstant layer that adds -4
    check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Layer Name

    Layer names are automatically assigned when an object of the same class is created, and ideally, they should be unique to distinguish different layers even if they are of the same type.

    By default, when you execute `AddConstant(3)`, an object of this layer is created, and it is given the name `AddConstant_i`, where `i` increments automatically as we create objects of the same `AddConstant` class.

    You can also specify the layer name manually to keep it fixed using the `name` parameter. For example, with `AddConstant(3, name="A layer that adds 3")`

    All layers must follow the convention of having a `name` parameter for the library to identify them. Also, in a given session, model names should be unique to simplify the implementation of some parts of the library.
    """)
    return


@app.cell
def _(nn):
    c1 = nn.AddConstant(3)
    print(c1.name)

    c2 = nn.AddConstant(3)
    print(c2.name)

    c3 = nn.AddConstant(3, name="My first layer :)")
    print(c3.name)
    return


if __name__ == "__main__":
    app.run()
