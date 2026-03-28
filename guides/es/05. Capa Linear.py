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
    # Capa Linear

    En este ejercicio debés implementar la capa `Linear`, que pesa las $I$ variables de entrada para generar $O$ valores de salida mediante la matriz de pesos $w$, de tamaño $I × O$.

    ## Caso con 1 entrada y 1 salida

    ## Caso con I entradas y S salidas

    En el caso más general, donde $w$ es una matriz que combina $I$ entradas de forma lineal para generar $O$ salidas, entonces $x \in R^{1×I}$ e $y \in R^{1×O}$. En este caso definimos entonces a $x$ e $y$ como _vectores fila_.
    $$
    x = \left( x_1, x_2, \dots, x_I \right)\\
    y = \left( y_1, y_2, \dots, y_O \right)
    $$

    Esta decisión es arbitraria: podrían definirse ambos como vectores columna, podríamos definir a $x$  como vector columna y a $y$ como fila, o viceversa. Dada la forma en que funcionan los frameworks, la definición como vector fila es la más usual, y entonces implica $w$ sea una matriz de tamaño $I×O$, y que la salida de la capa $y$ ahora se defina como:

    $$ y = x w$$

    Notamos que
    * $x w$ ahora es un producto matricial
    * En este caso es importante el orden entre $x$ y $w$, ya que el producto de matrices no es asociativo
    	* Un arreglo de $1×I$ ($x$) multiplicado por otro de $I×O$ ($w$) da como resultado un arreglo de $1×O$ ($y$)
    	* La definición inversa, $y=wx$, requeriría que $x$ e $y$ sean vectores columna, o que $w$ tenga tamaño $O×I$,

    ## Lotes

    Las capas reciben no un solo ejemplo, sino un lote de los mismos. Entonces, dada una entrada `x` de $N×I$ valores, donde $N$ es el tamaño de lote de ejemplos, `y` tiene tamaño $N×O$. El tamaño de $w$ no se ve afectado; sigue siendo $I×O$.

    Por ejemplo, si la entrada `x` es `[[1,-1]]` (tamaño $1×2$) y la capa `Linear` tiene como parámetros `w=[[2.0, 3.0],[4.0,5.0]]` (tamaño $2×2$), entonces la salida `y` será `x . w = [ [1,-1] . [2,4], [1,-1] . [3, 5] ] = [ 1*2+ (-1)*4, 1*3+ (-1)*5] = [-2, -2] `.

    Tu objetivo es implementar los métodos `forward` y `backward` de esta capa, de modo de poder utilizarla en una red neuronal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creación e Inicialización

    La capa `Linear` tiene un vector de parámetros `w`, que debe crearse en base a un tamaño de entrada y de salida de la capa, que debe establecerse al crearse.

    Respecto a la inicialización, lo usual es hacerlo con valores aleatorios. Para ello, deberás implementar la clase `initializers.RandomNormal`, que inicializa los parámetros con una normal de media 0 y una desviación estándar que se configura al crearse.
    """)
    return


@app.cell
def _(nn, utils):
    # Creamos una capa Linear con 2 valores de entrada y 3 de salida
    # inicializado con valores muestreados de una normal
    # con media 0 y desviación estándar 1e-12

    std=1e-12
    input_dimension=2
    output_dimension=3
    linear1=nn.Linear(input_dimension,output_dimension,initializer=nn.initializers.RandomNormal(std))
    print(f"Nombre de la capa: {linear1.name}")
    print(f"Parámetros de la capa : {linear1.get_parameters()}")
    print("(deben cambiar cada vez que vuelvas a correr esta celda)")
    print()

    linear2 = nn.Linear(input_dimension,output_dimension,initializer=nn.initializers.RandomNormal(std))

    w1 = linear1.get_parameters()["w"]
    w2 = linear2.get_parameters()["w"]


    print("Verificar que los pesos tengan media 0 y desviación std:")
    utils.check_mean(w1,0,tol=std)
    utils.check_mean(w2,0,tol=std)
    utils.check_std(w1,std,tol=std)
    utils.check_std(w2,std,tol=std)

    print("Verificar de que dos capas capas tienen valores iniciales de w distintos:")
    utils.check_different(w1,w2,tol=std/10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método forward

    Ahora que sabemos como crear e inicializar objetos de la capa `Linear`, comenzamos con el método `forward`, que podrás encontrar en el archivo `linear.py` de la carpeta `edunn/models`.

    Para verificar que la implementación de `forward` es correcta, utilizamos el inicializador `Constant`, pero luego por defecto la capa debe seguir utilizando un inicializador aleatorio como `RandomNormal`.
    """)
    return


@app.cell
def _(nn, np, utils):
    x = np.array([[3, -7], [-3, 7]])
    w = np.array([[2, 3, 4], [4, 5, 6]])
    initializer = nn.initializers.Constant(w)
    _layer = nn.Linear(2, 3, initializer=initializer)
    y = np.array([[-22, -26, -30], [22, 26, 30]])
    utils.check_same(y, _layer.forward(x))
    initializer = nn.initializers.Constant(-w)
    _layer = nn.Linear(2, 3, initializer=initializer)
    utils.check_same(-y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método backward

    En la implementación de la capa `Bias` las fórmulas de las derivadas eran relativamente sencillas, y la complejidad estaba más que todo en cómo utilizar el framework y comprender la diferencia entre la derivada de la entrada y la de los parámetros.

    El método backward de la capa `Linear` requiere calcular $\frac{δE}{δy}$ y $\frac{δE}{δw}$. En términos mecánicos, la implementación es muy similar a la de `Bias`, pero las fórmulas de las derivadas son más complicadas.

    Para no alargar demasiado este cuaderno, te dejamos [una explicación detallada del cálculo de las derivadas](http://facundoq.github.io/edunn/material/linear.html), tanto para $\frac{δE}{δx}$ como para $\frac{δE}{δw}$
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
    _layer = nn.Linear(features_in, features_out)
    # Test derivatives of a Linear layer with random values for `w`
    utils.check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


if __name__ == "__main__":
    app.run()
