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

    import edunn as nn
    import numpy as np

    return nn, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Capa Dense o FullyConnectd

    Las capas `Linear`, `Bias` y de activación (`Sigmoid`, `ReLU`, `TanH` y otras) suelen usarse en conjunto con la forma `dense(x) = activation(w*x+b)`, donde `activation` es alguna de estas funciones de activación. Esta capa suele denominarse `FullyConnected` o, como la llamaremos acá, `Dense`; el nombre viene de que cada salida de la capa depende de *todas* las entradas.

    En este ejercicio debés implementar la capa `Dense`, pero utilizando las capas `Linear`, `Bias` y de activaciones directamente, sin copiar su código.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creación e Inicialización

    La capa `Dense` debería tener un vector de parámetros `w`, otro vector `b` cada uno con un inicializador particular. Además, debería una función de activación.

    Pero para implementarla, vamos a utilizar tiene 3 *capas internas*: `Linear`, `Bias` y la de activación que llamaremos `Activation`. Por comodidad vamos también vamos a permitir especificar la activación con un string como `relu`, `sigmoid` o `tanh`.  En este caso, te ayudamos ya definiendo el constructor `__init__`, que asigna a las variables de instancia `self.linear`, `self.bias` y `self.activation` los objetos de las *capas internas* correspondientes y permite especificar los inicializadores de las mismas.

    También te ayudamos implementando `get_parameters`, que combina el diccionario de parámetros de cada subcapa en un gran diccionario único de gradientes de `Dense`.

    Te recomendamos que estudies el código de estos dos métodos (`__init__` y `get_parameters`) para ver como funcionan. Te ayudarán para luego implementar el `forward` y `backward` de `Dense`.
    """)
    return


@app.cell
def _(nn):
    # Creamos una capa Densa con 2 valores de entrada y 3 de salida
    # y activación ReLU
    # La capa lineal se inicializa de forma aleatoria
    # Mientras que la capa de bias en 0

    input_dimension=2
    output_dimension=3
    activation="relu"
    dense1=nn.Dense(input_dimension,output_dimension,
                     activation_name="relu",
                     linear_initializer=nn.initializers.RandomNormal(),
                     bias_initializer=nn.initializers.Constant(0),
                     )
    print(f"Nombre de la capa: {dense1.name}")
    print(f"Parámetro w de la capa: {dense1.get_parameters()['w']}")
    print("(debe cambiar cada vez que vuelvas a correr esta celda)")
    print(f"Parámetro b de la capa: {dense1.get_parameters()['b']}")
    print("(debe valer siempre 0)")
    print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método `forward`

    Ahora que sabemos como crear e inicializar objetos de la capa `Dense`, comenzamos con el método `forward`, que podrás encontrar en el archivo `dense.py` de la carpeta `edunn/models`.

    Para implementar el forward, deberás tomar el `x` de entrada y usarlo llamar al `forward` de las capas internas de tipo `Linear`, `Bias` y la `Activation`.

    Para verificar que la implementación de `forward` es correcta, utilizamos ambas veces el inicializador `Constant`, pero luego por defecto la capa sigue utilizando un inicializador aleatorio como `RandomNormal` por defecto.
    """)
    return


@app.cell
def _(nn, np):
    x = np.array([[3, -7], [-3, 7]])
    w = np.array([[2, 3, 4], [4, 5, 6]])
    b = np.array([1, 2, 3])
    linear_initializer = nn.initializers.Constant(w)
    bias_initializer = nn.initializers.Constant(b)
    _layer = nn.Dense(2, 3, linear_initializer=linear_initializer, bias_initializer=bias_initializer)
    y = np.array([[-21, -24, -27], [23, 28, 33]])
    nn.utils.check_same(y, _layer.forward(x))
    linear_initializer = nn.initializers.Constant(-w)
    bias_initializer = nn.initializers.Constant(-b)
    _layer = nn.Dense(2, 3, linear_initializer=linear_initializer, bias_initializer=bias_initializer)
    nn.utils.check_same(-y, _layer.forward(x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método `backward`

    Para implementar el `backward`, también deberás llamar al `backward` de las variables `self.linear`, `self.bias` y `self.activation` en el orden y forma correcta. Pista: es el contrario al del `forward`.

    En este caso, también te ayudamos combinando el diccionario de gradientes de cada capa en un gran diccionario único de gradientes de `Dense` utilizando el operador `**dict` que desarma un diccionario y `{**dict1, **dict2}` que los vuelve a combinar.
    """)
    return


@app.cell
def _(nn):
    samples = 100
    batch_size = 2
    features_in = 3
    features_out = 5
    input_shape = (batch_size, features_in)
    _layer = nn.Dense(features_in, features_out, activation_name='relu')
    # Test derivatives of a Dense layer with random values for `w`
    nn.utils.check_gradient.common_layer(_layer, input_shape, samples=samples)
    return


if __name__ == "__main__":
    app.run()
