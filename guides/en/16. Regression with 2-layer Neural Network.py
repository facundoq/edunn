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

    return (nn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2-Layer Neural Network for Regression

    Now that we have all the elements, we can define and train our first 2-layer neural network! We will also use it to estimate house prices using the [Boston Housing Dataset](https://www.kaggle.com/c/boston-housing).

    In this case, since the network is more powerful, we should expect a lower error compared to the previous linear regression model.

    You can also try other available datasets to load using `nn.datasets.load_regression`.
    """)
    return


@app.cell
def _(nn):
    dataset_name = 'boston'
    x, y = nn.datasets.load_regression(dataset_name)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    n, din = x.shape
    n, dout = y.shape
    print('Dataset sizes:', x.shape, y.shape)
    hidden_dim = 5
    model = nn.Sequential([nn.Dense(din, hidden_dim, activation_name='relu'), nn.Dense(hidden_dim, dout)])
    error = nn.MeanError(nn.SquaredError())
    # Network with two linear layers
    _optimizer = nn.GradientDescent(lr=0.01)
    _trainer = nn.SupervisedTrainer(model, _optimizer, error, epochs=1000, batch_size=32)
    _history = _trainer.train(x, y)
    # nn.plot.plot_history(_history, error_name=error.name)
    print('Model Error:')
    _y_pred = model.forward(x)
    nn.metrics.regression_summary(y, _y_pred)
    # Optimization algorithm
    nn.plot.regression1d_predictions(y, _y_pred)
    return din, dout, error, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparison with a Linear Regression Model

    As an additional verification, we will calculate the optimal parameters of a linear regression model and visualize the results. The error should be worse than that of the neural network (RMSE=3.27 or 3.28).
    """)
    return


@app.cell
def _(din, dout, error, nn, x, y):
    linear_model = nn.LinearRegression(din, dout)
    _optimizer = nn.GradientDescent(lr=0.01)
    _trainer = nn.SupervisedTrainer(linear_model, _optimizer, error, epochs=1000, batch_size=32)
    _history = _trainer.train(x, y)
    # nn.plot.plot_history(_history, error_name=error.name)
    _y_pred = linear_model.forward(x)
    print('Model Error:')
    nn.metrics.regression_summary(y, _y_pred)
    print()
    nn.plot.regression1d_predictions(y, _y_pred)
    return


if __name__ == "__main__":
    app.run()
