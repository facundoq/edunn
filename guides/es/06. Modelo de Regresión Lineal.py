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
    # Modelo de Regresión Lineal: `forward`

    Un modelo de Regresión Lineal se forma aplicando primero una capa lineal $f(x)=wx$ y luego una capa de bias $f(x)=x+b$ obteniendo finalmente la función $f(x)=wx+b$. Ahora bien, en lugar de ver a $f(x)$ como $wx+b$, podemos verlo como `x -> Linear -> Bias -> y`. Es decir, como una sucesión de capas, que cada una transforma la entrada `x` hasta obtener la salida `y`.

    En términos de código, el método `forward` de un modelo de `LinearRegression` es la composición de las funciones `forward` de las capas `Linear` y `Bias`:
    ````python
    y_linear = linear.forward(x)
    y = bias.forward(y_linear)
    ````

    Implementa el método `forward` del modelo `LinearRegression` en el archivo `edunn/models/linear_regression.py`:
    """)
    return


@app.cell
def _(nn, np, utils):
    x = np.array([[3, -7], [-3, 7]])
    w = np.array([[2, 3, 4], [4, 5, 6]])
    b = np.array([1, 2, 3])
    linear_initializer = nn.initializers.Constant(w)
    bias_initializer = nn.initializers.Constant(b)
    _layer = nn.LinearRegression(2, 3, linear_initializer=linear_initializer, bias_initializer=bias_initializer)
    y = np.array([[-21, -24, -27], [23, 28, 33]])
    utils.check_same(y, _layer.forward(x))
    linear_initializer = nn.initializers.Constant(-w)
    bias_initializer = nn.initializers.Constant(-b)
    _layer = nn.LinearRegression(2, 3, linear_initializer=linear_initializer, bias_initializer=bias_initializer)
    utils.check_same(-y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modelo de Regresión Lineal: `backward`

    El método `backward` de un modelo de `LinearRegression` es la composición *inversa* de las funciones `backward` de las capas `Linear` y `Bias`.

    Es decir, la derivada del error de la salida del modelo primero pasa por `Bias`, que calcula la derivada respecto de sus parámetros y lo devuelve en `δEδbias` (el único parámetro es $b$ en este caso). Ahora tenemos entonces `δEδx_bias` la derivada del error respecto de la salida de la capa `Linear`. Por ende, podemos hacer lo mismo pero al revés que el `forward`.

    ¡Este es el primer ejemplo (simple) de la aplicación del algoritmo de *backpropagation*! Esta vez, solo con dos modelos/capas. Luego lo generalizaremos con el modelo `Sequential`.

    En este caso, también te ayudamos combinando el diccionario de gradientes de cada capa en un gran diccionario único de gradientes de `LinearRegression` utilizando el operador `**` que desarma un diccionario, con `{**dict1, **dict2}` que los vuelve a combinar.
    """)
    return


@app.cell
def _(nn, utils):
    samples = 100
    batch_size = 2
    din = 3  # dimensión de entrada
    dout = 5  # dimensión de salida
    input_shape = (batch_size, din)
    _layer = nn.LinearRegression(din, dout)
    # Verificar las derivadas de un modelo de Regresión Lineal
    # con valores aleatorios de `w`, `b`, y `x`, la entrada
    utils.check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ¡Tenemos nuestro primer modelo complejo! Pero todavía no podemos entrenarlo. Para eso necesitamos:

    1. Un conjunto de datos
    2. Una función de error
    3. Un algoritmo de optimización para esa función de error

    En la siguientes guías, vamos a implementar (2) y luego (3), y tomando algún conjunto de datos (1) pondremos a prueba este modelo de Regresión Lineal.
    """)
    return


if __name__ == "__main__":
    app.run()
