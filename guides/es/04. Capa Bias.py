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
    # Capas Dense o Fully Connected o de Regresión Lineal

    La capa más común de una red es una capa que implementa la función `y = xw+b`, donde `x` es la entrada, `y` la salida, y `b` es un vector de sesgos y `w` es una matriz de pesos. No obstante, implementar esta capa puede ser difícil. En lugar de eso, separaremos la implementación en dos partes.

    * La capa `Bias`, que solo sumará `b` a su entrada, es decir `y=x+b`
    * La capa `Linear`, que sólo multiplicará a la entrada por la matriz de pesos `w`, es decir `y=w*x`
    * Combinando ambas capas, podremos recuperar la funcionalidad de la capa tradicional llamada `Dense` o `FullyConnected` en otras librerías, que por si sola nos permite resolver problemas con un modelo de regresión lineal con la salida `y=wx+b`.

    Comenzamos entonces con la capa `Bias`, la más simple de las dos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Capa Bias

    En este ejercicio debés implementar la capa `Bias`, que suma un valor distinto a cada una de sus entradas para generar su salida. Este valor _NO_ es constante, sino que esun parámetro de la red

    Por ejemplo, si la entrada `x` es `[3.5,-7.2]` y la capa `Bias` tiene como parámetros `[2.0, 3.0]`, entonces la salida `y` será `[5.5,-4.2]`.

    Tu objetivo es implementar los métodos `forward` y `backward` de esta capa, de modo de poder utilizarla en una red neuronal.

    Esta capa funciona para arreglos que tengan el mismo tamaño que los parámetros de la capa `Bias` (sin contar la dimensión de lote o batch)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creación e Inicialización

    La capa `Bias` tiene un vector de parámetros `b`, que debe crearse e inicializarse de alguna forma. Además, este parámetro se registra en la capa para poder se accedido posteriormente:

    ```python
    class Bias(Layer):
        def __init__(self,output_size:int,initializer:Initializer=Zero(),name=None):
                super().__init__(name=name)
                b = initializer.create( (output_size,))
                self.register_parameter("b", b)
    ```

    Vemos que al crear la capa podemos pasar un objeto de tipo `Initializer` que va a crear y asignarle el valor inicial al parámetro `b`. Por defecto, `b` se inicializa con ceros utilizando la clase `initializers.Zero`.

    ```python
    class Zero(Initializer):
        def initialize(self,p:np.ndarray):
            p[:]=0
    ```

    Examinando la implementación de esta clase, podemos ver que:
    * Hereda de Initializer
    * Implementa el método `initialize(self,p:np.ndarray)` que recibe un arreglo de numpy para inicializar
    * Utiliza `p[:]` para inicializar en 0 en lugar de `p=0`. Hay dos razones importantes para esto:
        * Utilizar `p=0` solo cambiaría la _variable local_ `p` en lugar de cambiar el _arreglo_ de numpy al cual `p` apunta
        * Al utilizar `p[:]` estamos cambiando el __contenido__ del arreglo de parámetros, que pertenece a una clase como `Bias` u otra que implementemos luego.

    Una vez creada la clase, podemos obtener el vector de parámetros `p` de la clase `Bias`con el método `get_parameters()`
    """)
    return


@app.cell
def _(nn):
    # Creamos una capa Bias con 2 valores de entrada/salida
    _bias = nn.Bias(2, initializer=nn.initializers.Zero())
    print(f'Nombre de la capa: {_bias.name}')
    print(f'Parámetros de la capa: {_bias.get_parameters()}')
    print()
    _bias2 = nn.Bias(2)
    # Por defecto, el inicializador ya es `Zero`
    print(f'Nombre de la capa: {_bias2.name}')
    print(f'Parámetros de la capa: {_bias.get_parameters()}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Acceso a los parámetros por nombre

    El método `get_parameters()` devuelve un diccionario de parámetros, ya que admite tener más de un parámetro por capa.

    Dado que ya sabemos en este caso cuál es el nombre del único parámetro de la capa, podemos acceder al mismo con su nombre en string `'b'`:
    """)
    return


@app.cell
def _(nn):
    # Creamos una capa Bias con 2 valores de entrada/salida
    _bias = nn.Bias(2, initializer=nn.initializers.Zero())
    print(f'Nombre de la capa: {_bias.name}')
    print(f"Parámetro 'b' de la capa: {_bias.get_parameters()['b']}")
    print()
    _bias2 = nn.Bias(2)
    # Por defecto, el inicializador ya es `Zero`
    print(f'Nombre de la capa: {_bias2.name}')
    print(f"Parámetro 'b' de la capa: {_bias2.get_parameters()['b']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Implementación de un inicializador constante

    Antes de comenzar con la implementación de la clase `Bias`, debés implementar el inicializador `Constant` que le asigna un valor o arreglo constante al parámetro, de modo que por ejemplo, se pueda inicializar `b` con todos valores `3` o con un vector de valores `[1,2,3,4]`.

    Buscá la clase `Constant` en el modulo `edunn/initializers.py` e implementa el método `initialize`.
    """)
    return


@app.cell
def _(nn, np, utils):
    # Creamos una capa Bias con 2 valores de salida (y también de entrada). 
    #Los parámetros están todos inicializados con 3
    valor = 3
    _bias = nn.Bias(2, initializer=nn.initializers.Constant(valor))
    print(f'Nombre de la capa: {_bias.name}')
    print(f"Parámetro 'b' de la capa: {_bias.get_parameters()['b']}")
    utils.check_same(_bias.get_parameters()['b'], np.array([3, 3]))
    print()
    valor = np.array([1, 2, 3, 4])
    _bias = nn.Bias(4, initializer=nn.initializers.Constant(valor))
    print(f'Nombre de la capa: {_bias.name}')
    # Creamos una capa Bias con valores iniciales  1,2,3,4
    print(f"Parámetro 'b' de la capa: {_bias.get_parameters()['b']}")
    utils.check_same(_bias.get_parameters()['b'], valor)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método forward

    Ahora que sabemos como crear e inicializar objetos de la capa `Bias`, comenzamos con el método `forward`, que podrás encontrar en el archivo `bias.py` de la carpeta `edunn/models`.

    Si los parámetros a sumar son $[b_1,...,b_f]$ y la entrada a la capa es $x = [x_1,x_2,...,x_f] $, entonces la salida $y$ es:

    $
    y([x_1,x2,...,x_f])= [x_1+b_1,x_2+b_2,...,x_n+b_f]
    $
    """)
    return


@app.cell
def _(nn, np, utils):
    x = np.array([[3.5, -7.2, 5.3], [-3.5, 7.2, -5.3]])
    _initializer = nn.initializers.Constant(np.array([2, 3, 4]))
    _layer = nn.Bias(3, initializer=_initializer)
    y = np.array([[5.5, -4.2, 9.3], [-1.5, 10.2, -1.3]])
    utils.check_same(y, _layer.forward(x))
    _initializer = nn.initializers.Constant(-np.array([2, 3, 4]))
    _layer = nn.Bias(3, initializer=_initializer)
    y = np.array([[1.5, -10.2, 1.3], [-5.5, 4.2, -9.3]])
    utils.check_same(y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método backward

    Además del cálculo de la salida de la capa, la misma debe poder propagar hacia atrás el gradiente del error de la red. Para eso, debés implementar el método `backward` que recibe $\frac{δE}{δy}$, es decir, las derivadas parciales del error respecto a la salida (gradiente) de esta capa , y devolver $\frac{δE}{δx}$, las derivadas parciales del error respecto de las entradas de esta capa.

    ## `δEδx`
    Para la capa `Bias` el cálculo del gradiente respecto de la entrada `δEδx` es simple, ya que es el mismo caso que con la capa `AddConstant`.

    $ \frac{δE}{δx} =\frac{δE}{δy} $

    No obstante, para esta capa también deberás implementar el gradiente con respecto a los parámetros `b`, de modo que se puedan optimizar para minimizar el error. Entonces también deberás calcular `δEδb`. Recordemos que:

    $
    y([x_1,x2,...,x_f])= [x_1+b_1,x_2+b_2,...,x_n+b_f]
    $

    Entonces, utilizando la regla de la cadena:
    $
    \frac{δE}{δb_i} = \frac{δE}{δy} \frac{δy_i}{δb_i} = \frac{δE}{δy_i} \frac{δy_i}{δb_i} = \frac{δE}{δy_i} \frac{δ x_i+b_i}{δb_i} = \frac{δE}{δy_i} 1 = \frac{δE}{δy_i}
    $

    Es decir,

    $ \frac{δE}{δx} =\frac{δE}{δy} $

    ## `δEδb`

    En el caso del gradiente del error con respecto a `b`, la fórmula es la misma, $ \frac{δE}{δb} =\frac{δE}{δy}$. Esto se debe a que $\frac{δy_i}{δb_i} = \frac{δ(x_i+b_i)}{δb_i} = \frac{δ(x_i+b_i)}{δx_i} =1$.

    Es decir, si vemos tanto a $b$ como a $x$ como entradas a la capa, $x+b$ es simétrico en $x$ y $b$ y por ende también lo son sus derivadas.
    """)
    return


@app.cell
def _(nn, np, utils):
    # Cantidad de valores aleatorios y tamaño de lote 
    # para generar valores de x y δEδy para la prueba de gradientes
    samples = 100
    batch_size = 2
    features = 4
    # Dimensiones de entrada y salida de la capa, e inicializador
    input_shape = (batch_size, features)
    _initializer = nn.initializers.Constant(np.array(range(features)))
    _layer = nn.Bias(features)
    utils.check_gradient.common_layer(_layer, input_shape, samples=samples)
    # Verificar derivadas de una capa Bias que con b=[0,1,2,3]
    _initializer = nn.initializers.Constant(-np.array(range(features)))
    _layer = nn.Bias(features)
    # Verificar derivadas de una capa Bias que con b=[0,-1,-2,-3]
    utils.check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


if __name__ == "__main__":
    app.run()
