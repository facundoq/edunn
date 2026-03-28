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
    # Linear Layer

    In this exercise, you need to implement the `Linear` layer, which weights the $I$ input variables to produce $O$ output values using the weight matrix $w$ of size $I × O$.

    ## Case with 1 Input and 1 Output

    In this case the math is similar to the well known 2D line equation $y = wx+b$. In this case $w$, $x$, $b$ and $y$ are all scalars, and we are just multiplying $x$ by $w$ and then adding $b$.

    ## Case with I Inputs and O Outputs

    In the most general case, where $w$ is a matrix that linearly combines $I$ inputs to generate $O$ outputs, then $x \in R^{1×I}$ and $y \in R^{1×O}$. In this case, we define both $x$ and $y$ as _row vectors_.
    $$
    x = \left( x_1, x_2, \dots, x_I \right)\\
    y = \left( y_1, y_2, \dots, y_O \right)
    $$

    This decision is arbitrary: we could define both as column vectors, we could define $x$ as a column vector and $y$ as a row vector, or vice versa. Given how frameworks typically work, defining them as row vectors is the most common, which implies that $w$ is a matrix of size $I×O$, and the output of the layer $y$ is defined as:

    $$ y = x w$$

    Note that:
    * $x w$ is now a matrix multiplication
    * The order between $x$ and $w$ matters because matrix multiplication is not associative
        * A $1×I$ array ($x$) multiplied by another $I×O$ array ($w$) results in a $1×O$ array ($y$)
        * The reverse definition, $y=wx$, would require that $x$ and $y$ be column vectors, or that $w$ has size $O×I$,

    ## Batches

    Layers receive not a single example, but a batch of examples.

    Given an input `x` of $N×I$ values, where $N$ is the batch size of examples, `y` has size $N×O$. The size of $w$ is not affected; it remains $I×O$, but now it has to work for multiple examples.

    For example, if the input `x` is `[[1,-1]]` (size $1×2$) and the `Linear` layer has parameters `w=[[2.0, 3.0],[4.0,5.0]]` (size $2×2$), then the output `y` will be `x . w = [ [1,-1] . [2,4], [1,-1] . [3, 5] ] = [ 1*2+ (-1)*4, 1*3+ (-1)*5] = [-2, -2] `.

    Your goal is to implement the `forward` and `backward` methods of this layer so that it can be used in a neural network.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creation and Initialization

    The `Linear` layer has a parameter vector `w` that must be created based on the input and output size of the layer, which should be set upon creation.

    Regarding initialization, it is common to initialize the weights with random values. To do this, you will need to implement the `initializers.RandomNormal` class, which initializes the parameters with a normal distribution with mean 0 and a standard deviation that is configured upon creation.
    """)
    return


@app.cell
def _(nn, utils):
    # Create a Linear layer with 2 input and 3 output values
    # Initialize it with values sampled from a normal distribution

    std = 1e-12
    input_dimension = 2
    output_dimension = 3
    linear1 = nn.Linear(input_dimension, output_dimension, initializer=nn.initializers.RandomNormal(std))
    print(f"Layer name: {linear1.name}")
    print(f"Layer parameters: {linear1.get_parameters()}")
    print("(these values should change each time you run this cell)")
    print()

    linear2 = nn.Linear(input_dimension, output_dimension, initializer=nn.initializers.RandomNormal(std))

    w1 = linear1.get_parameters()["w"]
    w2 = linear2.get_parameters()["w"]


    print("Check that weights have mean 0 and std deviation:")
    utils.check_mean(w1, 0, tol=std)
    utils.check_mean(w2, 0, tol=std)
    utils.check_std(w1, std, tol=std)
    utils.check_std(w2, std, tol=std)

    print("Check that two layers have different initial values for w:")
    utils.check_different(w1, w2, tol=std/10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forward Method

    Now that we know how to create and initialize `Linear` layer objects, let's move on to the `forward` method, which can be found in the `edunn/models/linear.py` file.

    To verify that the `forward` implementation is correct, we use the `Constant` initializer. However, by default the layer should use a random initializer like `RandomNormal`.
    """)
    return


@app.cell
def _(nn, np, utils):
    # create two inputs with 2 features
    x = np.array([[3, -7], [-3, 7]])
    w = np.array([[2, 3, 4], [4, 5, 6]])
    initializer = nn.initializers.Constant(w)
    _layer = nn.Linear(2, 3, initializer=initializer)
    y = np.array([[-22, -26, -30], [22, 26, 30]])
    # Initialize a 2x3 linear layer with specific weights
    utils.check_same(y, _layer.forward(x))
    initializer = nn.initializers.Constant(-w)
    _layer = nn.Linear(2, 3, initializer=initializer)
    # Check the result of the `forward`
    # Repeat the above with different weights
    utils.check_same(-y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backward Method

    In the implementation of the `Bias` layer, the derivative formulas were relatively simple, and the complexity was mostly in how to use the framework and understand the difference between the derivative with respect to the input and the derivative with respect to the parameters.

    The `backward` method of the `Linear` layer requires calculating $\frac{δE}{δy}$ and $\frac{δE}{δw}$. In terms of computational tools, the implementation is very similar to that of the `Bias` layer, but the derivative formulas are more complicated.

    To avoid making this notebook too long, we leave [a detailed explanation of the derivative calculations](http://facundoq.github.io/edunn/material/en/linear.html) for both $\frac{δE}{δx}$ and $\frac{δE}{δw}$. That'll help you implement the `backward` method for the `Linear` layer.
    """)
    return


@app.cell
def _(nn, utils):
    # Number of random values of x and δEδy to generate and test gradients
    samples = 100
    batch_size = 2
    features_in = 3
    features_out = 5
    input_shape = (batch_size, features_in)
    _layer = nn.Linear(features_in, features_out)
    # Test derivatives of a Linear layer with random values for `w`
    utils.check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


if __name__ == "__main__":
    app.run()
