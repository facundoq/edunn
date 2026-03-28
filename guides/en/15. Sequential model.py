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

    return nn, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `Sequential` Model for Neural Networks

    We have already implemented layers/models of all kinds: dense, activation functions, error layers, etc. Additionally, we have initializers and an optimizer based on stochastic gradient descent, as well as models that combine other layers like `LinearRegression` and `LogisticRegression`.

    To take the next step and define simple neural networks, we will implement the `Sequential` model. This model generalizes the ideas applied in `LinearRegression`, `LogisticRegression`, and `Dense`, allowing us to create a layer based on other layers. In previous cases, the layers to be used were predefined. `Sequential` will allow us to use any combination of layers we want.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creating a `Sequential` Model

    A `Sequential` model should be created with a list of other models/layers. This way, we specify what transformations and in what order will be performed to obtain the network's output.

    We can see several examples where we create a linear regression or logistic regression model or a Dense layer based on the `Sequential` model.

    `Sequential` also has a very useful method, `summary()`, which allows us to obtain a description of the layers and their parameters.
    """)
    return


@app.cell
def _(nn):
    din=5
    dout=3

    # Create a linear regression model
    layers = [nn.Linear(din,dout), nn.Bias(dout)]
    linear_regression = nn.Sequential(layers, name="Linear Regression")
    print(linear_regression.summary())


    # Create a linear regression model without the auxiliary variable `layers`
    linear_regression = nn.Sequential([nn.Linear(din,dout),
                                       nn.Bias(dout),
                                      ], name="Linear Regression")
    print(linear_regression.summary())

    # Create a logistic regression model
    logistic_regression = nn.Sequential([nn.Linear(din,dout),
                                       nn.Bias(dout),
                                       nn.Softmax(dout)
                                      ], name="Logistic Regression")
    print(logistic_regression.summary())


    # Create a Dense layer with ReLU activation
    dense_relu = nn.Sequential([nn.Linear(din,dout),
                               nn.Bias(dout),
                               nn.ReLU(dout)
                              ], name="Dense Layer with ReLU Activation")
    print(dense_relu.summary())
    return din, dout


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multi-Layer Networks with `Sequential`

    We will also create our first multi-layer neural networks by adding more layers to the model.
    """)
    return


@app.cell
def _(din, dout, nn):
    # Create a network with two Dense layers, both with internal dimensionality of 3
    network_layer2 = nn.Sequential([nn.Dense(din, 3, "relu"),
                                   nn.Dense(3, dout, "id")
                          ], name="Two-Layer Network")
    print(network_layer2.summary())



    # Create a network with 4 Dense layers
    # Internal dimensions are 2, 4, and 3
    # The final layer uses softmax activation
    network_layer4 = nn.Sequential([nn.Dense(din, 2, "relu"),
                                   nn.Dense(2, 4, "tanh"),
                                   nn.Dense(4, 3, "sigmoid"),
                                   nn.Dense(3, dout, "softmax"),
                          ], name="Four-Layer Network")
    print(network_layer4.summary())
    return network_layer2, network_layer4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `Sequential` Model Parameters

    The `Sequential` model also allows you to easily retrieve the parameters of all its internal models. For this purpose, we have already implemented the `get_parameters` method, which allows you to obtain _all_ the parameters of the internal models, but renamed so that if, for example, two models have the same parameter names, those names will not be repeated.
    """)
    return


@app.cell
def _(network_layer2, network_layer4):
    print("Parameter names of network_layer2")
    print(network_layer2.get_parameters().keys())

    print("Parameter names of network_layer4")
    print(network_layer4.get_parameters().keys())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `Sequential` `forward` Method

    Now, let's implement the `forward` method for `Sequential`. Given an input `x` and a sequence of models `M_1, M_2, ..., M_n` within `Sequential`, we must calculate the output `y` as:

    $$ y = M_n(...(M_2(M_1(x))...)$$

    In code terms, we need to iterate through the possible models (starting with the first one) and apply the `forward` method.

    ```python
    for m in models:
        x = m.forward(x)
    return x
    ```

    Implement the `forward` method for the `Sequential` class in `edunn/models/sequential.py`.
    """)
    return


@app.cell
def _(nn, np):
    x = np.array([[3, -7], [-3, 7]])
    w = np.array([[2, 3, 4], [4, 5, 6]])
    b = np.array([1, 2, 3])
    linear_initializer = nn.initializers.Constant(w)
    bias_initializer = nn.initializers.Constant(b)
    _layer = nn.Sequential([nn.Linear(2, 3, initializer=linear_initializer), nn.Bias(3, initializer=bias_initializer)])
    y = np.array([[-21, -24, -27], [23, 28, 33]])
    nn.utils.check_same(y, _layer.forward(x))
    linear_initializer = nn.initializers.Constant(-w)
    bias_initializer = nn.initializers.Constant(-b)
    _layer = nn.Sequential([nn.Linear(2, 3, initializer=linear_initializer), nn.Bias(3, initializer=bias_initializer)])
    nn.utils.check_same(-y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # `Sequential` `backward` Method

    Similar to the `Dense` layer, to implement the `backward` method, you should also call the `backward` method of each of the models in the reverse order compared to the forward pass. Given a tensor `δEδy` that contains the derivatives of the error with respect to each value of the output `y`, we need to calculate:
    * `δEδx`, the derivative of the error with respect to the input `x`
    * `δEδp_i`, the derivative of the error with respect to each parameter `p_i`

    To achieve this, we need to iterate through the possible models (starting with the last one) and apply the `backward` method, propagating the error backward and collecting the most important information, which is the derivatives of the error with respect to the parameters. In code terms,

    ```python
    δEδp = {}
    for m_i in reverse(models):
        δEδy, δEδp_i = m_i.backward(δEδy)
        add gradients of δEδp_i to δEδp
    return δEδy, δEδp
    ```

    In this case, we also provide the `merge_gradients` function, which you can call as `self.merge_gradients(layer, δEδp, gradients)`. This function allows you to add the parameters' `δEδp` of the layer `layer` to the final gradients dictionary `gradients` that should be returned.
    """)
    return


@app.cell
def _(nn):
    samples = 100
    batch_size = 2
    features_in = 3
    features_out = 5
    input_shape = (batch_size, features_in)
    _layer = nn.Sequential([nn.Linear(features_in, features_out), nn.Bias(features_out), nn.ReLU()])
    # Test derivatives of a Sequential model with random values for `w`
    nn.utils.check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Congratulations!

    You have implemented all the basic functions of a neural network library!

    Now, let's define some neural networks to improve performance compared to linear models (Linear Regression and Logistic Regression).
    """)
    return


if __name__ == "__main__":
    app.run()
