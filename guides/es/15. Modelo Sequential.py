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

    return nn, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modelo `Sequential` para Redes Neuronales

    Ya hemos implementado capas/modelos de todo tipo: densas, funciones de activación, de error, etc. Además, tenemos inicializadores, un optimizador basado en descenso de gradiente estocástico, y modelos que combinan otras capas como `LinearRegression` y `LogisticRegression`.

    Para dar el siguiente paso y poder definir redes neuronales simples, vamos a implementar el modelo `Sequential`. Este modelo generaliza las ideas aplicadas en `LinearRegression`, `LogisticRegression` y `Dense`, es decir, crear una capa en base a otras. En los casos anteriores, las capas a utilizar estaban predefinidas. `Sequential` nos permitirá utilizar cualquier combinación de capas que querramos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creación de un modelo `Sequential`

    Un modelo `Sequential` debe crearse con una lista de otros modelos/capas. De esta manera, específicaremos qué transformaciones y en qué orden se realizarán para obtener la salida de la red.

    Podemos ver varios ejemplos en donde creamos un modelo de regresión lineal, logística, o una capa Dense en base al modelo `Sequential`.

    `Sequential` también tiene un método muy útil, `summary()`, que nos permite obtener una descripción de las capas y sus parámetros.
    """)
    return


@app.cell
def _(nn):
    din=5
    dout=3

    # Creamos un modelo de regresión lineal 
    layers = [nn.Linear(din,dout), nn.Bias(dout)]
    linear_regression = nn.Sequential(layers,name="Regresión Lineal")
    print(linear_regression.summary())


    # Creamos un modelo de regresión lineal, pero sin la variable auxiliar `layers`
    linear_regression = nn.Sequential([nn.Linear(din,dout),
                           nn.Bias(dout),
                          ],name="Regresión Lineal")
    print(linear_regression.summary())

    # Creamos un modelo de regresión logística 
    logistic_regression = nn.Sequential([nn.Linear(din,dout),
                           nn.Bias(dout),
                           nn.Softmax(dout)
                          ],name="Regresión Logística")
    print(logistic_regression.summary())


    # Creamos un modelo tipo capa Dense con activación ReLU
    dense_relu = nn.Sequential([nn.Linear(din,dout),
                           nn.Bias(dout),
                           nn.ReLU(dout)
                          ],name="Capa tipo Dense con activación ReLU")
    print(dense_relu.summary())
    return din, dout


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Redes de varias capas con `Sequential`

    También vamos a crear nuestras primeras redes neuronales de varias capas, simplemente agregando más capas al modelo.
    """)
    return


@app.cell
def _(din, dout, nn):
    # Creamos una red con dos capas Dense, y una dimensionalidad de 3 interna
    network_layer2 = nn.Sequential([nn.Dense(din,3,"relu"),
                                   nn.Dense(3,dout,"id")
                          ],name="Red de dos capas")
    print(network_layer2.summary())



    # Creamos una red con 4 capas Dense
    # dimensiones internas de 2, 4 y 3
    # y función de activación final softmax
    network_layer4 = nn.Sequential([nn.Dense(din,2,"relu"),
                                   nn.Dense(2,4,"tanh"),
                                   nn.Dense(4,3,"sigmoid"),
                                   nn.Dense(3,dout,"softmax"),
                          ],name="Red de dos capas")
    print(network_layer4.summary())
    return network_layer2, network_layer4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Paramétros de `Sequential`

    El modelo `Sequential`  también permite obtener fácilmente los parámetros de todos sus modelos internos. Para eso ya hemos implementado el método `get_parameters` que permite obtener _todos_ los parámetros de los modelos internos, pero renombrados para que si, por ejemplo, dos modelos tienen el mismo nombre de sus parámetros, estos nombres no se repitan.
    """)
    return


@app.cell
def _(network_layer2, network_layer4):
    print("Nombres de los parámetros de network_layer2")
    print(network_layer2.get_parameters().keys())

    print("Nombres de los parámetros de network_layer4")
    print(network_layer4.get_parameters().keys())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método `forward` de `Sequential`

    Vamos ahora a implementar el método `forward` de `Sequential`. Para eso, dada una entrada `x`, y una sucesión de modelos `M_1,M_2,...,M_n` de `Sequential`, debemos calcular la salida `y` como:

    $$ y = M_n(...(M_2(M_1(x))...)$$

    En términos de código, debemos iterar por los posibles modelos (empezando por el primero) y aplicar el método `forward`

    ````python
    for m in models:
        x = m.forward(x)
    return x
    ````

    Implementá `forward` para la clase `Sequential` en `edunn/models/sequential.py`.
    """)
    return


@app.cell
def _(nn, np):
    x = np.array([[3, -7], [-3, 7]])
    w = np.array([[2, 3, 4], [4, 5, 6]])
    b = np.array([1, 2, 3])
    linear_initializer = nn.initializers.Constant(w)
    bias_initializer = nn.initializers.Constant(b)
    _layer = nn.Sequential([nn.Linear(2, 3, initializer=linear_initializer), nn.Bias(3, initializer=bias_initializer)])
    y = np.array([[-21, -24, -27], [23, 28, 33]])
    nn.utils.check_same(y, _layer.forward(x))
    linear_initializer = nn.initializers.Constant(-w)
    bias_initializer = nn.initializers.Constant(-b)
    _layer = nn.Sequential([nn.Linear(2, 3, initializer=linear_initializer), nn.Bias(3, initializer=bias_initializer)])
    nn.utils.check_same(-y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método `backward`

    Al igual que con `Dense`, para implementar el `backward`, también deberás llamar al `backward` de cada uno de los modelos en el orden _inverso_ al del forward. Dado un tensor `δEδy` que contiene las derivadas del error respecto a cada valor de la salida `y`, debemos calcular:
    * `δEδx`, la derivada del error respecto a la entrada `x`
    * `δEδp_i`, la derivada del error respecto a cada parámetro `p_i`

    Para ello, debemos iterar por los posibles modelos (empezando por el último) y aplicar el método `backward`, propagando el error para atrás, y recolectando en el proceso lo más importante, que son las derivadas del error respecto a los parámetros. En términos de código,

    ````python
    δEδp = {}
    for m_i in reverse(models):
        δEδy, δEδp_i = m_i.backward(δEδy)
        agregar los gradientes de δEδp_i a δEδp
    return δEδy,δEδp
    ````
    En este caso, también te ayudamos con la función `merge_gradients` que podés llamar como `self.merge_gradients(layer,δEδp,gradients)`. Esta función te permite agregar los parámetros `δEδp` de la capa `layer` al diccionario de gradientes final `gradients` que se debe retornar.
    """)
    return


@app.cell
def _(nn):
    samples = 100
    batch_size = 2
    features_in = 3
    features_out = 5
    input_shape = (batch_size, features_in)
    _layer = nn.Sequential([nn.Linear(features_in, features_out), nn.Bias(features_out), nn.ReLU()])
    # Test derivatives of a Sequential model with random values for `w`
    nn.utils.check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # !Felicitaciones!

    !Implementaste todas las funciones básicas de una librería de redes neuronales!

    Ahora vamos a definir algunas redes neuronales para mejorar el desempeño respecto de los modelos lineales (Regresión Lineal y Regresión Logística)
    """)
    return


if __name__ == "__main__":
    app.run()
