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
    # Red Neuronal de 2 Capas para Regresión

    Ahora que tenemos todos los elementos, podemos definir y entrenar nuestra primera red neuronal de 2 capas!. También la utilizaremos para estimar el precio de casas utilizando el conjunto de datos de [Casas de Boston](https://www.kaggle.com/c/boston-housing).

    En este caso, dado que la red es más potente, deberíamos obtener un error menor que el del modelo de regresión lineal anterior.

    Podés también probar otros conjuntos de datos disponibles para cargar con `nn.datasets.load_regression`
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
    #Red con dos capas lineales
    _optimizer = nn.GradientDescent(lr=0.01)
    _trainer = nn.SupervisedTrainer(model, _optimizer, error, epochs=1000, batch_size=32)
    _history = _trainer.train(x, y)
    # nn.plot.plot_history(_history, error_name=error.name)
    print('Error del modelo:')
    _y_pred = model.forward(x)
    nn.metrics.regression_summary(y, _y_pred)
    # Algoritmo de optimización
    nn.plot.regression1d_predictions(y, _y_pred)
    return din, dout, error, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparación con un modelo de Regresión Lineal

    Como verificación adicional, calcularemos los parámetros óptimos de un modelo de regresión lineal, y visualizamos los resultados. El error debería ser peor que de la red (RMSE=3.27 o 3.28).
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
    print('Error del modelo:')
    nn.metrics.regression_summary(y, _y_pred)
    print()
    nn.plot.regression1d_predictions(y, _y_pred)
    return


if __name__ == "__main__":
    app.run()
