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
    # Capa MultiplyConstant

    En este ejercicio debés implementar la capa `MultiplyConstant`,  que multiplica a cada una de sus entradas por un valor constante para generar su salida. Funciona de forma similar a `AddConstant`, pero en este caso multiplica en lugar de sumar, y por ende sus derivadas son ligeramente más complicadas.

    Por ejemplo, si la entrada `x` es `[3.5,-7.2,5.3]` y la capa `MultiplyConstant` se crea con la constante `2`, `y` será `[7.0,-14.4,10.6]`.

    Tu objetivo es implementar los métodos `forward` y `backward` de esta capa, de modo de poder utilizarla en una red neuronal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método forward

    El método `forward` calcula la salida `y` en base a la entrada `x`, como explicamos antes. En términos formales,  si la constante a sumar es $C$ y la entrada a la capa es $x = [x_1,x_2,...,x_n] $, entonces la salida $y$ es:

    $
    y([x_1,x2,...,x_n])= [x_1*C,x_2*C,...,x_n*C]
    $

    Comenzamos con el método `forward` de la clase `MultiplyConstant`, que podrás encontrar en el archivo `activations.py` de la carpeta `edunn/models`. Debés completar el código entre los comentarios:

    ```\"\"\" YOUR IMPLEMENTATION START \"\"\"```

    y

    ```\"\"\" YOUR IMPLEMENTATION END \"\"\"```

    Y luego verificar con la siguiente celda una capa que multiplica por 2 y otra que multiplica por -2. Si ambos chequeos son correctos, verás dos mensajes de <span style='background-color:green;color:white; '>éxito (success)</span>.
    """)
    return


@app.cell
def _(nn, np, utils):
    x = np.array([[3.5, -7.2, 5.3], [-3.5, 7.2, -5.3]])
    _layer = nn.MultiplyConstant(2)
    y = np.array([[7.0, -14.4, 10.6], [-7.0, 14.4, -10.6]])
    utils.check_same(y, _layer.forward(x))
    _layer = nn.MultiplyConstant(-2)
    y = -np.array([[7.0, -14.4, 10.6], [-7.0, 14.4, -10.6]])
    utils.check_same(y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método backward

    Además del cálculo de la salida de la capa, la misma debe poder propagar hacia atrás el gradiente del error de la red. Para eso, debés implementar el método `backward` que recibe $\frac{δE}{δy}$, es decir, las derivadas parciales del error respecto a la salida (gradiente) de esta capa , y devolver $\frac{δE}{δx}$, las derivadas parciales del error respecto de las entradas de esta capa.

    Para la capa `AddConstant` el cálculo del gradiente es fácil, ya que como:

    $
    y(x_1,x_2,...,x_n)= (x_1*C,x_2*C,...,x_n*C)
    $

    Entonces

    $y_i(x)=x_i*C$

    Y entonces

    $\frac{δE}{δx_i} = \frac{δE}{δy} * \frac{δy}{δx_i}$

    Como $y_i$ solo depende de $x_i$, podemos reescribir lo anterior como

    $ \frac{δE}{δx_i} = \frac{δE}{δy} * \frac{δy}{δx_i} = \frac{δE}{δy_i} * \frac{δy_i}{δx_i} $

    Dado que $y_i(x)=x_i*C$, entonces $\frac{δy_i}{δx_i} = C$ y por ende:
    $ \frac{δE}{δx_i} = \frac{δE}{δy} * \frac{δy}{δx_i} = \frac{δE}{δy_i} * C $

    Escribiendo entonces en forma vectorial para el vector x:

    $ \frac{δE}{δx} = [ \frac{δE}{δy_1} *C, \frac{δE}{δy_2} *C, ..., \frac{δE}{δy_n}*C ] = \frac{δE}{δy}*C $

    Con lo cual la capa simplemente propaga los gradientes de la capa siguiente, pero multiplicados por C.

    Completar el código en la función `backward` de la capa `MultiplyConstant` y verificar con la celda de abajo:
    """)
    return


@app.cell
def _(nn):
    from edunn.utils import check_gradient
    samples = 100
    input_shape = (5, 2)
    # number of random values of x and δEδy to generate and test gradients
    _layer = nn.MultiplyConstant(3)
    check_gradient.common_layer(_layer, input_shape, samples=samples)
    _layer = nn.MultiplyConstant(-4)
    check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


if __name__ == "__main__":
    app.run()
