import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    sys.path.insert(0, '../..')
    return


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
    # Capas Pooling

    La operación de _pooling_ ayuda a reducir la dimensionalidad espacial de los feature maps, así reduciendo la cantidad de parámetros de la red. Básicamente son convoluciones, donde el stride habitualmente es igual al tamaño del kernel y donde se calcula alguna función sobre todos los píxeles. Lo más usual es el máximo, el mínimo o el promedio. No solo reducen la dimensionalidad (así logrando un entrenamiento más rápido), sino que además generalmente ayudan en la clasificación, ya que permiten mayor eficacia al momento de generalizar.

    Una práctica comun para realizar una arquitectura convolucional es intercalar capas ``Conv2d`` con capas ``Pooling`` hasta llegar a capas ``Dense`` (Feed-Forward) que discriminarán las características aprendidas por las capas anteriores.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Max Pool
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Método Forward

    La operación de _Max Pooling_ es similar a la de convolución, pero sin necesidad de realizar una suma ponderada elemento a elemento, ya que en este caso se selecciona el máximo elemento de cada región particular encerrada por el kernel.

    La explicación de su cálculo es sencilla a partir de un ejemplo, supongamos que $x \in \mathbb{R}^{(4\times 4)}$ y $F=S=2$, de modo los $y_{ij}$ se definen como:

    $$
    \begin{array}{cc}
    y_{11} = \max ( x_{11},x_{12},x_{21},x_{22} ) & y_{12} = \max ( x_{13},x_{14},x_{23},x_{24} ) \\
    y_{21} = \max ( x_{31},x_{32},x_{41},x_{42} ) & y_{22} = \max ( x_{33},x_{34},x_{43},x_{44} )
    \end{array}
    $$

    Esto nos permite ver que el shape de salida de una capa Max Pool 2D está descripto por la misma ecuación que para el shape de salida de una capa convolucional.

    Implementa el método `forward` del modelo `MaxPool2d` en el archivo `edunn/models/pooling.py`.
    """)
    return


@app.cell
def _(nn, np):
    np.random.seed(123)

    x = np.random.rand(2,3,7,7)

    kernel_size,stride=2,2

    layer=nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    y=layer.forward(x)
    y, y.shape
    return kernel_size, layer, stride, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Método Backward

    Conociendo el gradiente de la función de error con respecto a los píxeles de salida, debemos calcular el gradiente total del error con respecto a los píxeles de entrada. Utilizando la regla de la cadena y derivadas parciales podemos llegar a la expresión:

    $$
    \frac{\partial E}{\partial x_{mn}}=\sum_{ij}\frac{\partial E}{\partial y_{ij}}\frac{\partial y_{ij}}{\partial x_{mn}}
    $$

    Tener en cuenta que las capas pooling no tienen parámetros, por lo que no hay que realizar el cálculo del gradiente del kernel.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `δE/δx`

    El cálculo de los gradientes del error $E$ con respecto al feature map de entrada $x$ se puede hacer derivando parcialmente como se explicó en guías anteriores.

    Continuando con el ejemplo previo, es evidente que $\frac{\partial y_{ij}}{\partial x_{mn}}$ es distinto de cero solo si $x_{mn}$ es el máximo elemento de la región de $y_{ij}$, de este modo sucedería que si $x_{12}$ es el máximo elemento de la región que abarca $y_{11}$, entonces:

    $$\frac{\partial y_{11}}{\partial x_{12}}=\frac{\partial x_{12}}{\partial x_{12}}=1$$

    Y el resto de derivadas con respecto a los otros $x_{mn}$ de la primera región serán cero. La derivada de $E$ con respecto a $x_{12}$ se formula multiplicando el gradiente local por los gradientes provenientes de la siguiente capa, quedando únicamente $\frac{\partial E}{\partial x_{12}}=\frac{\partial E}{\partial y_{11}} \cdot 1 \neq 0$ para la región de $y_{11}$.

    * El mismo razonamiento es válido para todas las  $(i,j)$  regiones.

    ![maxpool_backward.png](img/maxpool_backward.png)
    """)
    return


@app.cell
def _(layer, np, y):
    # Define el gradiente de la salida
    g = np.random.rand(*y.shape)

    # Propaga el gradiente hacia atrás a través de la convolución
    layer_grad = layer.backward(g)
    layer_grad
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comprobaciones con PyTorch
    """)
    return


@app.cell
def _():

    tnn = None
    torch = None
    return tnn, torch


@app.cell
def _(kernel_size, stride, tnn, torch, x):
    x_1 = torch.from_numpy(x).to(torch.double)
    _pool = tnn.MaxPool2d(kernel_size=kernel_size, stride=stride, return_indices=True)
    x_1.requires_grad = True
    y_torch, indices = _pool(x_1)
    return (y_torch,)


@app.cell
def _(utils, y, y_torch):
    utils.check_same(y_torch,y)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(nn, utils):
    _samples = 100
    _batch_size = 2
    _din = 3
    _dout = 5
    _input_shape = (_batch_size, _din, 11, 11)
    layer_1 = nn.MaxPool2d(kernel_size=3)
    utils.check_gradient.common_layer(layer_1, _input_shape, samples=_samples, tolerance=1e-05)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Avg Pool
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Método Forward

    La operación de _Average Pooling_ es similar a la de convolución, donde se calcula el promedio de los elementos que conforman a la región particular encerrada por el kernel.

    La explicación de su cálculo es sencilla a partir de un ejemplo, supongamos que $x \in \mathbb{R}^{(4\times 4)}$, $F=2$ y $S=1$, de modo los $y_{ij}$ se definen como:

    $$
    \begin{array}{ccc}
    y_{11} = \text{avg} ( x_{11},x_{12},x_{21},x_{22} ) &
    y_{12} = \text{avg} ( x_{12},x_{13},x_{22},x_{23} ) &
    y_{13} = \text{avg} ( x_{13},x_{14},x_{23},x_{24} ) \\

    y_{21} = \text{avg} ( x_{21},x_{22},x_{31},x_{32} ) &
    y_{22} = \text{avg} ( x_{22},x_{23},x_{32},x_{33} ) &
    y_{23} = \text{avg} ( x_{23},x_{24},x_{33},x_{34} ) \\

    y_{31} = \text{avg} ( x_{31},x_{32},x_{41},x_{42} ) &
    y_{32} = \text{avg} ( x_{32},x_{33},x_{42},x_{43} ) &
    y_{33} = \text{avg} ( x_{33},x_{34},x_{43},x_{44} )
    \end{array}
    $$

    Esto nos permite ver que el shape de salida de una capa Avg Pool 2D está descripto por la misma ecuación que para el shape de salida de una capa convolucional.

    Implementa el método `forward` del modelo `AvgPool2d` en el archivo `edunn/models/pooling.py`.

    > TIP: puedes reutilizar la función de forward de la capa `MaxPool2d` y cambiar `np.max` por `np.average`.
    """)
    return


@app.cell
def _(nn, np):
    np.random.seed(123)
    x_2 = np.random.rand(2, 3, 7, 7)
    kernel_size_1, stride_1 = (2, 1)
    # x = np.random.rand(1,1,4,4)
    layer_2 = nn.AvgPool2d(kernel_size=kernel_size_1, stride=stride_1)
    y_1 = layer_2.forward(x_2)
    (y_1, y_1.shape)
    return kernel_size_1, layer_2, stride_1, x_2, y_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Método Backward
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `δE/δx`

    El cálculo de los gradientes del error $E$ con respecto al feature map de entrada $x$ se puede hacer derivando parcialmente como se explicó en guías anteriores.

    <!-- Al tener que realizar el cálculo de la derivada de una matriz con respecto a otra, se considera cada elemento de la matriz como una variable independiente. Por lo tanto, la derivada de un elemento de la matriz $y$ con respecto a un elemento de la matriz $x$ considerará todas las posibles rutas por las que ese elemento de $x$ puede influir en el elemento de $y$.

    De este modo, $\frac{\partial y}{\partial x_{mn}} = \sum_{ij} \frac{\partial y_{ij}}{\partial x_{mn}}$, por la regla de la cadena para derivadas parciales, que nos indica que la derivada de una función con respecto a una variable es la suma de las derivadas de esa función a través de todas las rutas posibles (pudiendo contener términos que sean cero debido a la no dependencia de determinados $y_{ij}$ con $x_{mn}$). -->

    Continuando con el ejemplo previo, es evidente que $\frac{\partial y_{ij}}{\partial x_{mn}}$ es distinto de cero solo si $x_{mn}$ pertenece a la región de $y_{ij}$, donde además será igual a $\frac{1}{F_H \cdot F_W}$, entonces para calcular por ejemplo $\frac{\partial y}{\partial x_{12}}$:

    $$\frac{\partial y}{\partial x_{12}} = \frac{\partial y_{11}}{\partial x_{12}} + \frac{\partial y_{12}}{\partial x_{12}} + \frac{\partial y_{21}}{\partial x_{12}} + \frac{\partial y_{22}}{\partial x_{12}} = \frac{1}{4} + \frac{1}{4} + 0 + 0$$

    ya que:

    $$\begin{aligned}
    y_{11} &= \frac { x_{11}+\color{red}{x_{12}}\color{default}+x_{21}+x_{22} }{F_H \cdot F_W} \Rightarrow
        \frac{\partial y_{11}}{\partial x_{12}} = \frac{1}{4} \frac{\partial x_{12}}{\partial x_{12}} = \frac{1}{4} \\
    y_{12} &= \frac { \color{red}{x_{12}}\color{default}+x_{13}+x_{22}+x_{23} }{F_H \cdot F_W} \Rightarrow
        \frac{\partial y_{12}}{\partial x_{12}} = \frac{1}{4} \frac{\partial x_{12}}{\partial x_{12}} = \frac{1}{4} \\
    y_{21} &= \frac { x_{21}+x_{22}+x_{31}+x_{32} }{F_H \cdot F_W} \Rightarrow
        \frac{\partial y_{21}}{\partial x_{12}} = 0 \\
    y_{22} &= \frac { x_{22}+x_{23}+x_{32}+x_{33} }{F_H \cdot F_W} \Rightarrow
        \frac{\partial y_{22}}{\partial x_{12}} = 0
    \end{aligned}$$

    De este modo, el resto de derivadas con respecto a los otros $x_{mn}$ de la primera región serán **en un comienzo** iguales a $\frac{1}{F_H \cdot F_W}$ y según su influencia en la salida debido al valor de $S$ se irán incrementando en esa misma cantidad.

    Esto último mencionado se puede visualizar mejor si $S=2$, donde así es evidente que para la primera región, el resultado parcial de $\frac{\partial y}{\partial x_{mn}}$ para $m,n \in \{1,2\}$ ante la no dependencia entre regiones dadas por el kernel es:

    $$
    \begin{bmatrix}
    1/4 & 1/4 & 0 & 0 \\
    1/4 & 1/4 & 0 & 0 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0
    \end{bmatrix}
    $$

    * El mismo razonamiento es válido para todas las  $(i,j)$  regiones.
    """)
    return


@app.cell
def _(np):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    return


@app.cell
def _(layer_2, np, y_1):
    # Define el gradiente de la salida
    g_2 = np.random.rand(*y_1.shape)
    # g = np.ones_like(y)
    layer_grad_1 = layer_2.backward(g_2)
    # Propaga el gradiente hacia atrás a través de la convolución
    layer_grad_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comprobaciones con PyTorch
    """)
    return


@app.cell
def _(kernel_size_1, stride_1, tnn, torch, x_2):
    x_3 = torch.from_numpy(x_2).to(torch.double)
    _pool = tnn.AvgPool2d(kernel_size=kernel_size_1, stride=stride_1)
    x_3.requires_grad = True
    y_torch_1 = _pool(x_3)
    return (y_torch_1,)


@app.cell
def _(utils, y_1, y_torch_1):
    utils.check_same(y_torch_1, y_1)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(nn, utils):
    _samples = 100
    _batch_size = 2
    _din = 3
    _dout = 5
    _input_shape = (_batch_size, _din, 11, 11)
    layer_3 = nn.AvgPool2d(kernel_size=3, stride=3)
    utils.check_gradient.common_layer(layer_3, _input_shape, samples=_samples, tolerance=1e-05)
    return


if __name__ == "__main__":
    app.run()
