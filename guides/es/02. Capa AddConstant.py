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
    # Capa AddConstant

    En este ejercicio debés implementar la capa `AddConstant`, que agrega un valor constante a cada una de sus entradas para generar su salida.

    Por ejemplo, si la entrada `x` es `[3.5,-7.2,5.3]` y la capa AddConstant agrega el valor `3.0` a su entrada, entonces la salida `y` será `[6.5,-4.2,8.3]`.

    Tu objetivo es implementar los métodos `forward` y `backward` de esta capa, de modo de poder utilizarla en una red neuronal.

    Esta capa funciona para arreglos de entrada de cualquier tamaño, ya sean vectores, matrices, o arreglos con más dimensiones.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método forward

    El método `forward` calcula la salida `y` en base a la entrada `x`, como explicamos antes. En términos formales,  si la constante a sumar es $C$ y la entrada a la capa es $x = [x_1,x_2,...,x_n] $, entonces la salida $y$ es:

    $
    y([x_1,x2,...,x_n])= [x_1+C,x_2+C,...,x_n+C]
    $

    Comenzamos con el método `forward` de la clase `AddConstant`, que podrás encontrar en el archivo `activations.py` de la carpeta `edunn/models`. Debés completar el código entre los comentarios:

    ```\"\"\" YOUR IMPLEMENTATION START \"\"\"```

    y

    ```\"\"\" YOUR IMPLEMENTATION END \"\"\"```

    Y luego verificar con la siguiente celda una capa que suma 3 y otra que suma -3. Si ambos chequeos son correctos, verás dos mensajes de <span style='background-color:green;color:white; '>éxito (success)</span>.
    """)
    return


@app.cell
def _(nn, np, utils):
    x = np.array([[3.5, -7.2, 5.3], [-3.5, 7.2, -5.3]])
    _layer = nn.AddConstant(3)
    y = np.array([[6.5, -4.2, 8.3], [-0.5, 10.2, -2.3]])
    utils.check_same(y, _layer.forward(x))
    _layer = nn.AddConstant(-3)
    y = np.array([[0.5, -10.2, 2.3], [-6.5, 4.2, -8.3]])
    utils.check_same(y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método backward

    Además del cálculo de la salida de la capa, la misma debe poder propagar hacia atrás el gradiente del error de la red. Para eso, debés implementar el método `backward` que recibe $\frac{δE}{δy}$, es decir, las derivadas parciales del error respecto a la salida (gradiente) de esta capa , y devolver $\frac{δE}{δx}$, las derivadas parciales del error respecto de las entradas de esta capa.

    Para la capa `AddConstant` el cálculo del gradiente de la salida $y$ respecto de la entrada $x$ es fácil, pero recordemos que tenemos que devolver el gradiente de $E$ respecto de $x$. Recordemos entonces la forma de la salida:

    $
    y([x_1,x_2,...,x_n])= [x_1+C,x_2+C,...,x_n+C]
    $

    Nuestro objetivo es calcular $\frac{δE}{δx}$. Para eso, nos enfocaremos en $\frac{δE}{δx_i}$, es decir la derivada del error respecto de una entrada en particular.
    Luego, aplicando regla de la cadena, podemos escribir $\frac{δE}{δx_i}$ como:

    $\frac{δE}{δx_i} = \frac{δE}{δy} * \frac{δy}{δx_i} = \sum_j \frac{δE}{δy_j} * \frac{δy_j}{δx_i} $

    Nos conviene entonces ver cada elemento de la salida escrito como:

    $y_i(x)=x_i+C$

    Y por ende:

    $\frac{δy_i}{δx_i} = 1 + 0  =1$

    Como no hay interacción entre elementos de distinto índice, es decir, $y_i$ solo depende de $x_i$, en la sumatoria anterior si $i\neq j$  entonces $\frac{δy_j}{δx_i}=0$. Por eso podemos quitar la sumatoria y ahora utilizar la regla de la cadena solo con $y_i$:

    $\frac{δE}{δx_i} = \frac{δE}{δy} * \frac{δy}{δx_i} = \sum_j \frac{δE}{δy_j} * \frac{δy_j}{δx_i} =  \frac{δE}{δy_i} * \frac{δy_i}{δx_i} $

    Sabiendo que  $\frac{δy_i}{δx_i} = 1$

    $ \frac{δE}{δx_i} = \frac{δE}{δy} * \frac{δy}{δx_i} = \frac{δE}{δy_i} * 1 = \frac{δE}{δy_i} $

    Escribiendo entonces en forma vectorial para el vector x:

    $ \frac{δE}{δx} = [ \frac{δE}{δy_1}, \frac{δE}{δy_2}, ..., \frac{δE}{δy_n} ] = \frac{δE}{δy} $

    Con lo cual la capa simplemente propaga los gradientes de la capa siguiente.

    Notar que por simplicidad, en el código llamamos a estos vectores `δEδy` y `δEδx`. También aclaramos que en este caso $C$ es una constante y NO un parámetro de la red, por lo cual no debemos calcular $\frac{δE}{δC}$.

    La verificación del gradiente se hace con la función `check_gradient_layer_random_sample`. Esta función genera muestras aleatorias de `x` y de `δEδy`, y luego compara el gradiente analítico (tu implementación) contra el gradiente numérico.

    El gradiente numérico _aproxima_ las derivadas parciales utilizando la fórmula de la derivada $\frac{δf(x)}{δx}= \lim_{h→0} \frac{f(x+h)-f(x)}{h}$ con un valor de $h$ muy pequeño ($h=10^{-12}$). En realidad, para una una mejor aproximación, utiliza la derivada _centrada_, cuya fórmula es $\frac{δf(x)}{δx}_{aprox}= \frac{f(x+h)-f(x-h)}{2h}$. Esta técnica de _verificación de gradientes_ es una técnica estándar para comprobar la implementación correcta de una red neuronal.

    Completar el código en la función `backward` de la capa `AddConstant` entre los comentarios:

    ```\"\"\" YOUR IMPLEMENTATION START \"\"\"```

    y

    ```\"\"\" YOUR IMPLEMENTATION END \"\"\"```

    Y luego verificar con la siguiente celda una capa que suma 3 y otra que suma -3. Si ambos chequeos son correctos, verás dos mensajes de <span style='background-color:green;color:white; '>éxito (success)</span>.
    """)
    return


@app.cell
def _(nn):
    from edunn.utils import check_gradient
    samples = 100
    input_shape = (5, 2)
    # number of random values of x and δEδy to generate and test gradients
    _layer = nn.AddConstant(3)
    check_gradient.common_layer(_layer, input_shape, samples=samples)
    _layer = nn.AddConstant(-4)
    # Test derivatives of an AddConstant layer that adds 3
    # Test derivatives of an AddConstant layer that adds -4
    check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Nombre de capa

    Los nombres de la capa se asignan de forma automática al crearse un objeto de la misma, e idealmente deben ser únicos para poder diferenciar distintas capas aunque sean del mismo tipo.

    Por defecto, al ejecutar `AddConstant(3)` se crea un objeto de esta capa, y se le pone el nombre de la `AddConstant_i` donde `i` va incrementándose automáticamente a medida que creamos objetos de la misma clase `AddConstant`.

    También se puede especificar el nombre de la capa manualmente, para que quede fijo, utilizando el parámetro `name`, por ejemplo con `AddConstant(3,name="Una capa que suma 3")`

    Todas las capas deben seguir la convención de tener un parámetro `name` para que la librería las identifique.
    """)
    return


@app.cell
def _(nn):
    c1 = nn.AddConstant(3)
    print(c1.name)

    c2 = nn.AddConstant(3)
    print(c2.name)

    c3 = nn.AddConstant(3, name="Mi primera capa :)")
    print(c3.name)
    return


if __name__ == "__main__":
    app.run()
