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
    # Capa Softmax

    Para problemas de clasificación, es útil tener un modelo que pueda generar distribuciones de probabilidad como salida. De esta forma, para un problema de `C` clases, el modelo puede tener como salida un vector de `C` elementos `f(x)=y` donde cada `y_i` es un valor entre 0 y 1 que indica la probabilidad de que el ejemplo `x` pertenezca a la clase `i`. Además, como `y` es una distribución', tenemos que $\sum_{i=1}^C y_i =1$,

    Un modelo de regresión lineal puede generar un vector de `C` elementos con los _puntajes_ de cada clase, pero estos valores estarán en el rango $(-\infty, +\infty)$, con lo cual nunca podrían cumplir las propiedades de una distribución de probabilidad que mencionamos arriba.

    En este ejercicio debés implementar la capa `Softmax`, que justamente dado un vector `x` de `C` puntajes por clase, lo convierte en un vector `y` de probabilidades por clase. Para eso, implementa la función Softmax, con $y =(y_1,y_2,...,y_C) = Softmax((x_1,x_2,...,x_C)) = Softmax(x)$, donde cada $y_i$ es la probabilidad de la clase $i$.  Entonces, por ejemplo, dados puntajes de clase `[-5,100,100]`, la función `Softmax` generará las probabilidades `[0,0.5,0.5]`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método forward

    El método `forward` utiliza la siguiente fórmula para `y`:

    $$y=
    \frac{[e^{x_1},...,e^{x_c}]}{e^{x_1}+...+e^{x_c}} =
    \frac{[e^{x_1},...,e^{x_c}]}{N}$$

    O visto elemento por elemento, cada valor de $y$ se define como:
    $$y_i(x) =  \frac{e^{x_i}}{e^{x_1}+...+e^{x_c}} $$

    Aquí, utilizamos la función exponencial ($e^x$) para convertir cada puntaje del vector `x` del rango $(-\infty, +\infty)$ al rango $(0, +\infty)$, ya que la función exponencial tiene esa imagen.

    Además, $e^x$ es monótonamente creciente en $x$, entonces a valores mayores de $x$, valores mayores de $e^x$, con lo cual si un puntaje es alto, también lo será la probabilidad y viceversa.

    Ahora bien, además cada elemento se divido por el valor $N$, que sirve para normalizar los valores, obteniendo:
    1. Valores entre 0 y 1
    2. Que la suma de valores sea 1

    Es decir, los axiomas de una distribución de probabilidad como mencionamos anteriormente

    Implementá el método `forward` de la clase `Softmax`.
    """)
    return


@app.cell
def _(nn, np, utils):
    x = np.array([[0, 0, 100], [0, 100, 0.0], [100, 100, 0.0], [50, 50, 0.0], [1, 1, 1]], dtype=float)
    _layer = nn.Softmax()
    y = np.array([[0, 0, 1], [0, 1, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]], dtype=float)
    utils.check_same(y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método backward

    El método `backward` de la función softmax requiere varios pasos, ya que debido a la normalización cada salida de la softmax depende de cada entrada.

    Para no hacer tan largo este cuaderno, los detalles del cálculo de la derivada están en un  [apunte en línea](http://facundoq.github.io/edunn/material/softmax_derivada.html).

    Implementá el método `backward` de la clase `Softmax`:
    """)
    return


@app.cell
def _(nn):
    from edunn.utils import check_gradient
    samples = 100
    input_shape = (5, 2)
    # number of random values of x and δEδy to generate and test gradients
    _layer = nn.Softmax()
    # Test derivatives of an AddConstant layer that adds 3
    check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


if __name__ == "__main__":
    app.run()
