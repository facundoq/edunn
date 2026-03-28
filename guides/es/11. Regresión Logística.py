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
    # Modelo de Regresión Logística: `forward`

    Un modelo de Regresión Logística se forma aplicando la función `Softmax` a un modelo de Regresión Lineal. De esta forma, esta función convierte al vector de salida de la Regresión Lineal en un vector que representa una distribución de probabilidad.

    La función de la Regresión logística es $f(x)=softmax(wx+b)$. No obstante, como hicimos con el modelo `LinearRegression`, podemos ver este modelo como la aplicación de
    * Una capa `Linear` $f(x)=wx$,
    * Una capa `Bias` $f(x)=x+b$
    * Una capa `Softmax` $f(x)=softmax(x)$

    Es decir, tenemos la siguiente secuencia de transformaciońes `x → Linear → Bias → Softmax → y`.

    Implementa el método `forward` del modelo `LogisticRegression` en el archivo `edunn/models/logistic_regression.py`. Para eso, ya definimos e inicializamos modelos internos de clase `Linear`, `Bias` y `Softmax`; solo tenés que llamar a sus `forward`s respectivos en el orden adecuado.
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
    # Modelo de Regresión Logística: `backward`

    El método `backward` de un modelo de `LogisticRegression` es la composición *inversa* de las funciones `backward` de las capas `Linear`, `Bias`, y `Softmax`. Recordá que estas se aplican en el orden contrario al método forward.

    En este caso, también te ayudamos combinando el diccionario de gradientes de cada capa en un gran diccionario único de gradientes de `LogisticRegression` utilizando el operador `**` que desarma un diccionario, con `{**dict1, **dict2}` que los vuelve a combinar.

    Implementá el método `backward` del modelo `LogisticRegression`:
    """)
    return


@app.cell
def _(nn, utils):
    samples = 100
    batch_size = 2
    _din = 3  # dimensión de entrada
    _dout = 5  # dimensión de salida
    input_shape = (batch_size, _din)
    _layer = nn.LogisticRegression(_din, _dout)
    # Verificar las derivadas de un modelo de Regresión Logística
    # con valores aleatorios de `w`, `b`, y `x`, la entrada
    utils.check_gradient.common_layer(_layer, input_shape, samples=samples, tolerance=1e-05)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regresión Logística aplicada

    Ahora que tenemos todos los elementos, podemos definir y entrenar nuestro primer modelo regresión logística para clasificar las flores del conjunto de datos de [Iris](https://www.kaggle.com/uciml/iris).

    En este caso, vamos a entrenar el modelo con la función de error cuadrático medio; no obstante, si bien esta forma del error funcionará para este problema, hace el que problema de optimización no sea _convexo_ y por ende no haya un único mínimo global. Más adelante, implementaremos la función de error de _Entropía Cruzada_, diseñada específicamente para lidiar con salidas que representan distribuciones de probabilidad.
    """)
    return


@app.cell
def _(nn):
    from edunn import metrics, datasets
    _x, _y, classes = datasets.load_classification('iris', onehot=True)
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
    print('Error del modelo:')
    y_pred = model.forward(_x)
    metrics.classification_summary_onehot(_y, y_pred)
    return


if __name__ == "__main__":
    app.run()
