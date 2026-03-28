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
    # Capa de Error Cuadrático

    En este ejercicio debés implementar la capa de error `SquaredError`, que permite calcular el error de un lote de ejemplos.

    Las capas de error son diferentes a las capas normales por dos motivos:

    1. No solo tienen como entrada la salida de la capa anterior, sino también el valor esperado de la capa anterior (`y` e `y_true`).
    2. Para un lote de $n$ ejemplos, su salida es un vector de tamaño $n$. Es decir, indican el valor del error de cada ejemplo con un escalar (número real).

    La capa de error también debe poder realizar la operación `backward` de modo de poder propagar hacia atrás el gradiente del error a la red.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método forward

    El método `forward` de la capa `SquaredError` simplemente debe calcular la distancia euclídea al cuadrado entre `y`, los valores producidos por la red, e `y_true`, el valor esperado por la misma.

    Por ejemplo, si $y=[2,-2]$ y $y_{true}=[3,3]$, entonces la salida de la capa es:

    $$E(y,y_{true})=d_2(y,y_{true})=d_2([2,-2],[3,3])=(2-3)^2+(-2-3)^2 = 1^2+(-5)^2=1+25=26$$

    En general, dados dos vectores $a=[a_1,\dots,a_n]$ y $b=[b_1,\dots,b_n]$, la distancia euclídea al cuadrado $d_2$ es:

    $$
    d_2(a,b)= d_2([a_1,\dots,a_n],[b_1,\dots,b_n]) =(a_1-b_1)^2+\dots+(a_n-b_n)^2
    $$

    En el caso de un lote de ejemplos, el cálculo es independiente para cada ejemplo. Es importante entonces que la suma de las diferencias al cuadrado se haga por cada ejemplo (fila) y no por cada característica (columna).
    """)
    return


@app.cell
def _(nn, np, utils):
    y = np.array([[2, -2], [-4, 4]])
    y_true = np.array([[3, 3], [-5, 2]])
    _layer = nn.SquaredError()
    E = np.array([[26], [5]])
    utils.check_same(E, _layer.forward(y, y_true))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método backward

    Ahora puedes calcular el error de una red, bien! Esta es la capa final de la red cuando se está entrenando. Por ende el método backward de una capa de error no recibe $\frac{δE}{δy}$; de hecho, debe calcularlo directamente a partir de $y$, $y_{true}$, y la definición del error. Además tampoco hay parámetros.

    Por ende, en este caso, la derivada es simple. Solo debemos calcular $\frac{δE}{δy}$, la derivada del error respecto a la salida calculada por la red, $y$.
    En este caso $E$ es simétrico respecto de sus entradas, así que llamemosla nuevamente $a$ y $b$, y entonces calculemos la derivada respecto del elemento $i$ de $a$ (la de $b$ sería igual):

    $$
    \frac{δE(a,b)}{δa_i} = \frac{δ((a_1-b_1)^2+\dots+(a_n-b_n)^2)}{δa_i} \\
    = \frac{δ((a_i-b_i)^2)}{δa_i} = 2 (a_i-b_i) \frac{δ((a_i-b_i))}{δa_i} \\
    = 2 (a_i-b_i) 1 = 2 (a_i-b_i)
    $$
    Generalizando para todo el vector $a$, entonces:
    $$
    \frac{δE(a,b)}{δa} = 2 (a-b)
    $$
    Donde $a-b$ es una resta entre vectores.

    Nuevamente, como este error es por cada ejemplo, entonces los cálculos son independientes en cada fila.
    """)
    return


@app.cell
def _(nn, utils):
    # number of random values of x and δEδy to generate and test gradients
    samples = 100
    batch_size = 2
    features_in = 3
    features_out = 5
    input_shape = (batch_size, features_in)
    _layer = nn.SquaredError()
    utils.check_gradient.squared_error(_layer, input_shape, samples=samples)
    return


if __name__ == "__main__":
    app.run()
