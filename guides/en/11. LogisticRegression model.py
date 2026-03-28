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
    # Logistic Regression Model: `forward`

    A Logistic Regression model is formed by applying the `Softmax` function to a Linear Regression model. This function converts the output vector of Linear Regression into a vector representing a probability distribution.

    The function for Logistic Regression is $f(x) = softmax(wx + b)$. However, as we did with the `LinearRegression` model, we can view this model as the application of:
    * A `Linear` layer $f(x) = wx$,
    * A `Bias` layer $f(x) = x + b$,
    * A `Softmax` layer $f(x) = softmax(x)$.

    In other words, we have the following sequence of transformations: `x → Linear → Bias → Softmax → y`.

    Implement the `forward` method of the `LogisticRegression` model in the `edunn/models/logistic_regression.py` file. For this, we have already defined and initialized internal class models `Linear`, `Bias`, and `Softmax`; you just need to call their respective `forward` methods in the correct order.
    """)
    return


@app.cell
def _(nn, np, utils):
    _x = np.array([[1, 0], [0, 1], [1, 1]])
    w = np.array([[100, 0, 0], [0, 100, 0]])
    b = np.array([0, 0, 0])
    linear_initializer = nn.initializers.Constant(w)
    bias_initializer = nn.initializers.Constant(b)
    _layer = nn.LogisticRegression(2, 3, linear_initializer=linear_initializer, bias_initializer=bias_initializer)
    _y = np.array([[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]])
    utils.check_same(_y, _layer.forward(_x))
    _y = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0, 0, 1]])
    linear_initializer = nn.initializers.Constant(-w)
    bias_initializer = nn.initializers.Constant(-b)
    _layer = nn.LogisticRegression(2, 3, linear_initializer=linear_initializer, bias_initializer=bias_initializer)
    utils.check_same(_y, _layer.forward(_x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Logistic Regression Model: `backward`

    The `backward` method of a `LogisticRegression` model is the *inverse* composition of the `backward` methods of the `Linear`, `Bias`, and `Softmax` layers. Remember that these are applied in the reverse order compared to the `forward` method.

    In this case, we also help you by combining the gradient dictionaries of each layer into a single large gradient dictionary for `LogisticRegression` using the `**` python operator to unpack and repack dictionaries with `{**dict1, **dict2}`.

    Implement the `backward` method of the `LogisticRegression` model:
    """)
    return


@app.cell
def _(nn, utils):
    samples = 100
    batch_size = 2
    _din = 3  # input dimension
    _dout = 5  # output dimension
    input_shape = (batch_size, _din)
    _layer = nn.LogisticRegression(_din, _dout)
    # Check the derivatives of a Logistic Regression model with random values of `w`, `b`, and `x`, the input
    utils.check_gradient.common_layer(_layer, input_shape, samples=samples, tolerance=1e-05)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Applied Logistic Regression

    Now that we have all necessary ingredients, we can define and train our first Logistic Regression model to classify the flowers in the [Iris dataset](https://www.kaggle.com/uciml/iris).

    In this case, we will train the model with the mean squared error function. However, while this form of error will work for this problem, it makes the optimization problem non-convex, and therefore, there is no unique global minimum.

    Later, we will implement the Cross-Entropy error function, designed specifically to deal with outputs that represent probability distributions. For now, let's just use `SquaredError`, knowing it is suboptimal.
    """)
    return


@app.cell
def _(nn):
    _x, _y, classes = nn.datasets.load_classification('iris', onehot=True)
    _x = (_x - _x.mean(axis=0)) / _x.std(axis=0)
    n, _din = _x.shape
    n, _dout = _y.shape
    print('Dataset sizes:', _x.shape, _y.shape)
    model = nn.LogisticRegression(_din, _dout)
    error = nn.MeanError(nn.SquaredError())
    optimizer = nn.GradientDescent(lr=0.1)
    trainer = nn.SupervisedTrainer(model, optimizer, error, epochs=1000, batch_size=32)
    history = trainer.train(_x, _y)
    # nn.plot.plot_history(history, error_name=error.name)
    print('Model Error:')
    y_pred = model.forward(_x)
    nn.metrics.classification_summary_onehot(_y, y_pred)
    return


if __name__ == "__main__":
    app.run()
