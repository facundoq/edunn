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
    # Descenso de gradiente

    El descenso de gradiente es una técnica de optimización simple pero efectiva para entrenar modelos derivables.

    En cada iteración del algoritmo, se calcula la derivada del error respecto a cada uno de los parámetros `δEδp`, actualizan los pesos en la dirección contraria al gradiente. Esta actualización está mediada por el parámetro `α` que indica la tasa de aprendizaje.

    El algoritmo de descenso de gradiente es simple:

    ```python
    for i in range(iteraciones):
        for p in model.parameters()
            # usamos p[:] para modificar los valores de p
            # y no crear una nueva variable
            p[:] = p - α * δEδp(x,y)
    ```

    Este pseudocódigo obvia algunas partes engorrosas. En particular, la iteración sobre los valores de entrada `x` y salida `y` de los ejemplos, en su versión por `batches`, y el cálculo del error y las derivadas `δEδp`.

    La librería `edunn` cuenta con la clase `BatchedGradientOptimizer` que se encarga de eso, y nos permite implementar un optimizador de forma muy simple creando una subclase de ella, e implementando el método `optimize_batch`, en donde solo tenemos preocuparnos por optimizar el modelo utilizando las derivadas calculadas con un batch del conjunto de datos.

    Para este ejercicio, hemos creado la clase `GradientDescent`, que subclasifica a `BatchedGradientOptimizer`. Implemente, entonces, la parte crucial del método `optimize_batch` de `GradientDescent`, para que actualice los parámetros en base a los los gradientes ya calculados.

    Para probar este optimizador, vamos a utilizar un modelo falso y error falso que nos permitan controlar de manera la entrada al optimizador. La flexiblidad de la clase `Model` de `edunn` permite hacer esto muy fácilmente creando las clases `FakeModel` y `FakeError`, que ignoran realmente sus entradas y salidas, y solo sirven para que `FakeModel` inicialice 2 parámetros con valore 0 y retorne `[-1,1]` como derivada para ellos.
    """)
    return


@app.cell
def _(nn, np, utils):
    #Modelo falso con un vector de parámetros con valor inicial [0,0] y gradientes que siempre son [1,-11]
    _model = nn.FakeModel(parameter=np.array([0, 0]), gradient=np.array([1, -1]))
    # función de error falso cuyo error es siempre 1 y las derivadas también
    _error = nn.FakeError(error=1, derivative_value=1)
    fake_samples = 3
    # Conjunto de datos falso, que no se utilizará realmente
    fake_x = np.random.rand(fake_samples, 10)
    fake_y = np.random.rand(fake_samples, 5)
    _optimizer = nn.GradientDescent(lr=2)
    _trainer = nn.SupervisedTrainer(_model, _optimizer, _error, verbose=False, epochs=1, batch_size=fake_samples)
    # Optimizar el modelo por 1 época con lr=2
    _history = _trainer.train(fake_x, fake_y)
    expected_parameters = np.array([-2, 2])
    utils.check_same(expected_parameters, _model.get_parameters()['parameter'])
    _trainer = nn.SupervisedTrainer(_model, _optimizer, _error, verbose=False, epochs=100, batch_size=32)
    _history = _trainer.train(fake_x, fake_y)
    expected_parameters = np.array([-4, 4])
    # Optimizar el modelo por 1 época *adicional* con lr=2
    utils.check_same(expected_parameters, _model.get_parameters()['parameter'])
    _optimizer = nn.GradientDescent(lr=1)
    _trainer = nn.SupervisedTrainer(_model, _optimizer, _error, verbose=False, epochs=3, batch_size=fake_samples)
    _history = _trainer.train(fake_x, fake_y)
    expected_parameters = np.array([-7, 7])
    # Optimizar el modelo por 3 épocas más, ahora con con lr=1    
    utils.check_same(expected_parameters, _model.get_parameters()['parameter'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Entrenamiento de un modelo de Regresión Lineal con Descenso de gradiente

    Ahora que tenemos todos los elementos, podemos definir y entrenar nuestro primer modelo `RegresionLineal` para estimar el precio de casas utilizando el conjunto de datos de [Casas de Boston](https://www.kaggle.com/c/boston-housing)
    """)
    return


@app.cell
def _(nn):
    from edunn import metrics, datasets
    x, y = datasets.load_regression('boston')
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    n, din = x.shape
    n, dout = y.shape
    print('Dataset sizes:', x.shape, y.shape)
    _model = nn.LinearRegression(din, dout)
    _error = nn.MeanError(nn.SquaredError())
    _optimizer = nn.GradientDescent(lr=0.001)
    _trainer = nn.SupervisedTrainer(_model, _optimizer, _error, epochs=1000, batch_size=32)
    _history = _trainer.train(x, y)
    # nn.plot.plot_history(_history, error_name=_error.name)
    print('Error del modelo:')
    _y_pred = _model.forward(x)
    metrics.regression_summary(y, _y_pred)
    nn.plot.regression1d_predictions(y, _y_pred)
    return metrics, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparación con sklearn

    Como verificación adicional, calcularemos los parámetros óptimos de un modelo de regresión lineal con sklearn, y visualizamos los resultados. El error debería ser similar al de nuestro modelo (RMSE=3.27 o 3.28).
    """)
    return


@app.cell
def _(metrics, nn, x, y):
    from sklearn import linear_model
    _model = linear_model.LinearRegression()
    _model.fit(x, y)
    _y_pred = _model.predict(x)
    print('Error del modelo:')
    metrics.regression_summary(y, _y_pred)
    print()
    nn.plot.regression1d_predictions(y, _y_pred)
    return


if __name__ == "__main__":
    app.run()
