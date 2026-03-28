import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Note:** This guide is currently in Spanish. An English translation is pending.
    """)
    return


@app.cell
def _():
    import sys
    sys.path.insert(0, '../..')
    return (sys,)


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    import numpy as np
    import edunn as nn
    from edunn import utils

    return nn, np, utils


@app.cell
def _(np, sys):
    np.set_printoptions(threshold=sys.maxsize)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Capa Convolucional 1D

    La convolución es una operación matemática fundamental en el procesamiento de señales y sistemas. Se utiliza ampliamente para el filtrado de señales, el suavizado de datos y la detección de características.

    En el contexto del procesamiento de señales, la convolución 1D se utiliza para extraer ciertas características de la señal, como bordes en el caso de las imágenes, o para eliminar ruido.

    $$
    y[n] = (x \circledast w) = \sum_{k} w[k] \cdot x[n+k]
    $$

    > NOTA: a cotinuación se explica cómo realizar la capa ``Conv2d``, donde su versión en 1D se puede interpretar como un caso particular de ella, de modo que la dimensión de la altura es siempre 1 e implica que el _stride_ y _padding_ sean en ese caso 1 y 0 respectivamente.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Capa Convolucional 2D

    Siguiendo la misma idea, podemos extender el concepto de convolución sobre matrices. Es decir, una convolución en 2 dimensiones. Esto nos sirve para imágenes en escala de grises, RGB y por lotes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Método Forward

    Una capa convolucional 2D se forma aplicando una operación de convolución 2D a la entrada, seguida de una adición de Bias si se desea. La convolución es una operación sobre dos funciones $x$ y $w$, que produce una tercera función que puede ser interpretada como una versión de la entrada $x$ "filtrada" por $w$ (también conocido como kernel).

    * Si bien la convolución se define en forma continua, a nosotros nos interesa su versión discreta.

    Entonces, dado un batch de imágenes de entrada de shape $(N,C_{\text{in}},H,W)$ y un batch de filtros de shape $(M,C_{\text{out}},F,F)$, la operación de convolución se puede expresar como:

    $$
    y[l,m,i,j] = (x \circledast w) = \sum_{a}\sum_{b} w[m,:,a,b] \cdot x[l,:,i+a,j+b]
    $$

    Donde $N$ es la cantidad de imágenes del lote, $C_{\text{in/out}}$ es el número de canales de la imagen de entrada y feature map de salida (respectivamente), $M$ es la cantidad de feature maps deseados y $F$ es el tamaño del kernel.

    En esta fórmula se ve que las sumas se realizan sobre las dimensiones del kernel y sobre los $C$ canales a la vez, además se asume que el **stride** (paso) es de 1 y el **padding** (relleno) es de 0. El término de bias se suma después de la operación de convolución.

    * Las funciones de activación pueden funcionar de manera matricial, aplicándose a cada valor de $y[k,i,j]$ sin necesidad de cambio alguno.

    |||
    |:-:|:-:|
    |![conv2d_forward.gif](img/conv2d_forward.gif)|![conv2d_multilayer.gif](img/conv2d_multilayer.gif)|
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hiperparámetros

    El tamaño de la matriz resultante $y$ y está estrictamente relacionado por los siguientes parámetros:

    * `kernel_size`: es el tamaño del filtro utilizado.
    * `stride`: es el número de saltos que da el filtro cada vez que se aplica.
    * `padding`: es la cantidad de píxeles rellenos con cero en los bordes.
       * Aplicar el filtro de forma discreta ocasiona dos problemas:
           * Pérdida de información en los bordes.
           * Reducción del tamaño final del vector.
       <!-- * Se utiliza para obtener una imagen resultante de un tamaño buscado. -->

    |strides|padding|
    |:-:|:-:|
    |![conv2d_strides.gif](img/conv2d_strides.gif)|![conv2d_padding.gif](img/conv2d_padding.gif)|

    ### Shapes

    El tamaño de la salida $y$ (es decir, los feature maps resultantes) depende del tamaño $H_{in} \times W_{in}$ de la imagen de entrada, el tamaño del kernel $F$, el stride (paso) $S$ y el padding (relleno) $P$. La fórmula general para calcular la altura $H_{out}$ y el ancho $W_{out}$ de la salida en una capa convolucional es:

    $$
    \begin{aligned}
    A_{out} &= \left\lfloor \frac{A_{in} - F + 2P}{S} \right\rfloor + 1 & \text{donde $A \in \{W, H\}$} \\
    \end{aligned}
    $$

    Por lo tanto, el shape de la salida $y$ es $(N, M, H_{out}, W_{out})$. El shape del bias depende únicamente del número de feature maps $M$. Notar que:

    - El tamaño del batch en la salida es igual al tamaño del batch en la entrada. Esto se debe a que cada imagen en el batch se procesa de forma independiente.
    - El tamaño de los canales en la salida es igual al tamaño del batch en el kernel. Esto se debe a que cada canal de salida se genera al convolucionar la entrada con un kernel diferente.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Implementación

    Similarmente como sucede con el Modelo Lineal, en lugar de ver a $y$ como $x \circledast w + b$, podemos verlo como `x -> Convolution -> Bias -> y`. Es decir, como una sucesión de capas, donde cada una transforma la entrada `x` hasta obtener la salida `y`.

    En términos de código, el método `forward` de un modelo de `Convolution` es la composición de las funciones `forward` de las capas `Convolution` y `Bias`:

    ```python
    y_conv = conv2d(x)  # Implementar la función dentro del mismo modelo
    y = bias.forward(y_conv)
    ```

    > TIP: la operación de convolución también se puede representar como la **cross-correlation** entre $x$ y $\text{rot}_{180^\circ } \left \{ w \right \}$, es decir:
    > $$
    y = \left( \begin{bmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \end{bmatrix}  \circledast \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix} \right)
    \Leftrightarrow
    \left( \begin{bmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \end{bmatrix} \star \begin{bmatrix} w_{22} & w_{21} \\ w_{12} & w_{11} \end{bmatrix} \right) = y
    $$

    Implementa el método `forward` del modelo `Convolution` en el archivo `edunn/models/convolution.py`.
    """)
    return


@app.cell
def _(nn, np):
    np.random.seed(123)
    x = np.random.rand(2, 3, 7, 7)
    w = np.random.rand(4, 3, 5, 5)
    stride, padding = (2, 1)
    _kernel_initializer = nn.initializers.Constant(w)
    layer = nn.Conv2d(x.shape[1], w.shape[0], kernel_size=w.shape[2:], stride=stride, padding=padding, kernel_initializer=_kernel_initializer)
    return layer, padding, stride, w, x


@app.cell
def _(layer, x):
    y = layer.forward(x)
    y, y.shape
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Método Backward

    Conociendo el gradiente de la función de error con respecto a los píxeles de salida, debemos calcular el gradiente total del error con respecto a los píxeles de entrada y los pesos de los filtros. Utilizando la regla de la cadena y derivadas parciales podemos llegar a las expresiones:

    $$
    \frac{\partial E}{\partial x_{mn}}=\sum_{ij}\frac{\partial E}{\partial y_{ij}}\frac{\partial y_{ij}}{\partial x_{mn}}
    \quad\quad\quad
    \frac{\partial E}{\partial w_{mn}}=\sum_{ij}\frac{\partial E}{\partial y_{ij}}\frac{\partial y_{ij}}{\partial w_{mn}}
    $$

    Donde cada píxel $x_{mn}$ de la entrada y del filtro $w_{mn}$ contribuye a uno o más píxeles $y_{ij}$ de salida, siendo solo los píxeles de salida que aparecen en las ecuaciones anteriores para un dado $x$ o $w$ los que contribuyen durante el cálculo del forward.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SIN stride o padding

    La explicación de su cálculo es sencilla a partir de un ejemplo, supongamos que $x \in \mathbb{R}^{(3\times 3)}$ y $w \in \mathbb{R}^{(2\times 2)}$, de modo los $y_{ij}$ se definen como:

    $$\begin{aligned}
    y_{11} &= x_{11}w_{11} + x_{12}w_{12} + x_{21}w_{21} + x_{22}w_{22} \\
    y_{12} &= x_{12}w_{11} + x_{13}w_{12} + x_{22}w_{21} + x_{23}w_{22} \\
    y_{21} &= x_{21}w_{11} + x_{22}w_{12} + x_{31}w_{21} + x_{32}w_{22} \\
    y_{22} &= x_{22}w_{11} + x_{23}w_{12} + x_{32}w_{21} + x_{33}w_{22}
    \end{aligned}$$

    #### `δE/δw`

    El cálculo de los gradientes del error $E$ con respecto al filtro $w$ (vector de pesos aprendido) se puede hacer derivando parcialmente como se explicó en guías anteriores:

    $$
    \begin{aligned}
    \frac{\partial E}{\partial w_{11}} &= \frac{\partial E}{\partial y_{11}} \frac{\partial y_{11}}{\partial w_{11}} + \frac{\partial E}{\partial y_{12}} \frac{\partial y_{12}}{\partial w_{11}} + \frac{\partial E}{\partial y_{21}} \frac{\partial y_{21}}{\partial w_{11}} + \frac{\partial E}{\partial y_{22}} \frac{\partial y_{22}}{\partial w_{11}}
    \\
    \frac{\partial E}{\partial w_{12}} &= \frac{\partial E}{\partial y_{11}} \frac{\partial y_{11}}{\partial w_{12}} + \frac{\partial E}{\partial y_{12}} \frac{\partial y_{12}}{\partial w_{12}} + \frac{\partial E}{\partial y_{21}} \frac{\partial y_{21}}{\partial w_{12}} + \frac{\partial E}{\partial y_{22}} \frac{\partial y_{22}}{\partial w_{12}}
    \\
    \frac{\partial E}{\partial w_{21}} &= \frac{\partial E}{\partial y_{11}} \frac{\partial y_{11}}{\partial w_{21}} + \frac{\partial E}{\partial y_{12}} \frac{\partial y_{12}}{\partial w_{21}} + \frac{\partial E}{\partial y_{21}} \frac{\partial y_{21}}{\partial w_{21}} + \frac{\partial E}{\partial y_{22}} \frac{\partial y_{22}}{\partial w_{21}}
    \\
    \frac{\partial E}{\partial w_{22}} &= \frac{\partial E}{\partial y_{11}} \frac{\partial y_{11}}{\partial w_{22}} + \frac{\partial E}{\partial y_{12}} \frac{\partial y_{12}}{\partial w_{22}} + \frac{\partial E}{\partial y_{21}} \frac{\partial y_{21}}{\partial w_{22}} + \frac{\partial E}{\partial y_{22}} \frac{\partial y_{22}}{\partial w_{22}}
    \end{aligned}
    $$

    Donde utilizando las ecuaciones anteriormente definidas al principio de esta celda, es fácil ver que:

    $$
    \begin{aligned}
    \frac{\partial E}{\partial w_{11}} &= \frac{\partial E}{\partial y_{11}} x_{11} + \frac{\partial E}{\partial y_{12}} x_{12} + \frac{\partial E}{\partial y_{21}} x_{21} + \frac{\partial E}{\partial y_{22}} x_{22}
    \\
    \frac{\partial E}{\partial w_{12}} &= \frac{\partial E}{\partial y_{11}} x_{12} + \frac{\partial E}{\partial y_{12}} x_{13} + \frac{\partial E}{\partial y_{21}} x_{22} + \frac{\partial E}{\partial y_{22}} x_{23}
    \\
    \frac{\partial E}{\partial w_{21}} &= \frac{\partial E}{\partial y_{11}} x_{21} + \frac{\partial E}{\partial y_{12}} x_{22} + \frac{\partial E}{\partial y_{21}} x_{31} + \frac{\partial E}{\partial y_{22}} x_{32}
    \\
    \frac{\partial E}{\partial w_{22}} &= \frac{\partial E}{\partial y_{11}} x_{22} + \frac{\partial E}{\partial y_{12}} x_{23} + \frac{\partial E}{\partial y_{21}} x_{32} + \frac{\partial E}{\partial y_{22}} x_{33}
    \end{aligned}
    $$

    Tiene un patrón idéntico al de una convolución entre $x$ y $\frac{\delta E}{\delta y}$:

    $$
    \frac{\delta E}{\delta w} = \begin{bmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \end{bmatrix}  \circledast \begin{bmatrix} \frac{\partial E}{\partial y_{11}} & \frac{\partial E}{\partial y_{12}} \\ \frac{\partial E}{\partial y_{21}} & \frac{\partial E}{\partial y_{22}} \end{bmatrix} = x \circledast \frac{\delta E}{\delta y}
    $$

    #### `δE/δx`

    El razonamiento para el cálculo de los gradientes del $E$ con respecto a la imagen de entrada $x$ es el mismo que antes:

    $$
    \begin{aligned}
    \frac{\partial E}{\partial x_{11}}&=\frac{\partial E}{\partial y_{11}}w_{11}+\frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{22}}\;\:\, 0 \;
    \\
    \frac{\partial E}{\partial x_{12}}&=\frac{\partial E}{\partial y_{11}}w_{12}+\frac{\partial E}{\partial y_{12}}w_{11}+\frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{22}}\;\:\, 0 \;
    \\
    \frac{\partial E}{\partial x_{13}}&=\frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{12}}w_{12}+\frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{22}}\;\:\, 0 \;
    \\
    \frac{\partial E}{\partial x_{21}}&=\frac{\partial E}{\partial y_{11}}w_{21}+\frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{21}}w_{11}+\frac{\partial E}{\partial y_{22}}\;\:\, 0 \;
    \\
    \frac{\partial E}{\partial x_{22}}&=\frac{\partial E}{\partial y_{11}}w_{22}+\frac{\partial E}{\partial y_{12}}w_{21}+\frac{\partial E}{\partial y_{21}}w_{12}+\frac{\partial E}{\partial y_{22}}w_{11}
    \\
    \frac{\partial E}{\partial x_{23}}&=\frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{12}}w_{22}+\frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{22}}w_{11}
    \\
    \frac{\partial E}{\partial x_{31}}&=\frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{21}}w_{21}+\frac{\partial E}{\partial y_{22}}\;\:\, 0 \;
    \\
    \frac{\partial E}{\partial x_{32}}&=\frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{21}}w_{22}+\frac{\partial E}{\partial y_{22}}w_{21}
    \\
    \frac{\partial E}{\partial x_{33}}&=\frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+\frac{\partial E}{\partial y_{22}}w_{22}
    \end{aligned}
    $$

    Tal cálculo se puede automatizar mediante el operador de **full-convolution**, en la que se aplica padding a la entrada de tal manera que la dimensión de la salida sea la misma que la de la entrada. Además, para poder representarlo de tal manera, es obligatorio que la matriz de filtros $w$ se rote 180°.

    $$
    \frac{\delta E}{\delta x} = \begin{bmatrix} \frac{\partial E}{\partial y_{11}} & \frac{\partial E}{\partial y_{12}} \\ \frac{\partial E}{\partial y_{21}} & \frac{\partial E}{\partial y_{22}} \end{bmatrix} \circledast \begin{bmatrix} w_{22} & w_{21} \\ w_{12} & w_{11} \end{bmatrix} = \left. \left[ \frac{\delta E}{\delta y} \circledast \text{rot}_{180^\circ } \left \{ w \right \} \right] \right|_{\text{full-padding=F-1}}
    $$

    ![conv2d_backward.gif](img/conv2d_backward.gif)

    La fórmula para lograr el padding adecuado es fácil deducirla y queda como ejercicio para el lector demostrarla (puede encontrar ayuda en el blog [cs231n](https://cs231n.github.io/convolutional-networks/)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CON stride y padding

    La explicación de su cálculo es sencilla a partir de un ejemplo, supongamos que $x \in \mathbb{R}^{(5\times 5)}$ y $w \in \mathbb{R}^{(3\times 3)}$, con $S=2$ de modo los $y_{ij}$ se definen como:

    $$\begin{aligned}
    y_{11} &=x_{11}w_{11}+x_{12}w_{12}+x_{13}w_{13}+x_{21}w_{21}+x_{22}w_{22}+x_{23}w_{23}+x_{31}w_{31}+x_{32}w_{32}+x_{33}w_{33} \\
    y_{12} &=x_{13}w_{11}+x_{14}w_{12}+x_{15}w_{13}+x_{23}w_{21}+x_{24}w_{22}+x_{25}w_{23}+x_{33}w_{31}+x_{34}w_{32}+x_{35}w_{33} \\
    y_{21} &=x_{31}w_{11}+x_{32}w_{12}+x_{33}w_{13}+x_{41}w_{21}+x_{42}w_{22}+x_{43}w_{23}+x_{51}w_{31}+x_{52}w_{32}+x_{53}w_{33} \\
    y_{22} &=x_{33}w_{11}+x_{34}w_{12}+x_{35}w_{13}+x_{43}w_{21}+x_{44}w_{22}+x_{45}w_{23}+x_{53}w_{31}+x_{54}w_{32}+x_{55}w_{33}
    \end{aligned}$$

    #### `δE/δx`

    Debido al tamaño de la entrada, la cantidad de derivadas parciales de $E$ respecto de los $x_{mn}$ es sustancialmente mayor a la de ejemplos previos, por lo tanto se deja oculta su demostración quedando en el lector si quiere incurrir en el _inhóspito_ cálculo de las mismas.

    <details><summary>Demostración</summary>

    $$
    \begin{aligned}
    \frac{\partial E}{\partial x_{11}}&=
        \frac{\partial E}{\partial y_{11}}w_{11}+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{12}}&=
        \frac{\partial E}{\partial y_{11}}w_{12}+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{13}}&=
        \frac{\partial E}{\partial y_{11}}w_{13}+
        \frac{\partial E}{\partial y_{12}}w_{11}+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{14}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}w_{12}+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{15}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}w_{13}+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{21}}&=
        \frac{\partial E}{\partial y_{11}}w_{21}+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{22}}&=
        \frac{\partial E}{\partial y_{11}}w_{22}+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{23}}&=
        \frac{\partial E}{\partial y_{11}}w_{23}+
        \frac{\partial E}{\partial y_{12}}w_{21}+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{24}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}w_{22}+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{25}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}w_{23}+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{31}}&=
        \frac{\partial E}{\partial y_{11}}w_{31}+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}w_{11}+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{32}}&=
        \frac{\partial E}{\partial y_{11}}w_{32}+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}w_{12}+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{33}}&=
        \frac{\partial E}{\partial y_{11}}w_{33}+
        \frac{\partial E}{\partial y_{12}}w_{31}+
        \frac{\partial E}{\partial y_{21}}w_{13}+
        \frac{\partial E}{\partial y_{22}}w_{11} \\
    \frac{\partial E}{\partial x_{34}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}w_{32}+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}w_{12} \\
    \frac{\partial E}{\partial x_{35}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}w_{33}+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}w_{13} \\
    \frac{\partial E}{\partial x_{41}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}w_{21}+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{42}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}w_{22}+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{43}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}w_{23}+
        \frac{\partial E}{\partial y_{22}}w_{21} \\
    \frac{\partial E}{\partial x_{44}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}w_{22} \\
    \frac{\partial E}{\partial x_{45}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}w_{23} \\
    \frac{\partial E}{\partial x_{51}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}w_{31}+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{52}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}w_{32}+
        \frac{\partial E}{\partial y_{22}}\;\:\, 0 \; \\
    \frac{\partial E}{\partial x_{53}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}w_{33}+
        \frac{\partial E}{\partial y_{22}}w_{31} \\
    \frac{\partial E}{\partial x_{54}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}w_{32} \\
    \frac{\partial E}{\partial x_{55}}&=
        \frac{\partial E}{\partial y_{11}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{12}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{21}}\;\:\, 0 \;+
        \frac{\partial E}{\partial y_{22}}w_{33}
    \end{aligned}
    $$

    </details>

    Tales resultados siguen un patrón común, que es fácil ver si **se dilata a $\frac{\partial E}{\partial y}$ en una cantidad igual a $S-1$**, se la rellena con ceros para obtener una full-convolution y se rota el filtro $w$ en 180°.

    ![conv2d_dilate.gif](img/conv2d_dilate.gif)

    #### `δE/δw`

    Este caso tiene gran similitud al anterior en cuanto a cómo tratarlo, y el patrón se basa también en una convolución donde se debe **dilatar a $\frac{\partial E}{\partial y}$ en una cantidad igual a $S-1$**, sin necesidad alguna de realizar padding.

    $$
    \begin{aligned}
    \frac{\partial E}{\partial w_{11}}&=\frac{\partial E}{\partial y_{11}}x_{11}+\frac{\partial E}{\partial y_{12}}x_{13}+\frac{\partial E}{\partial y_{21}}x_{31}+\frac{\partial E}{\partial y_{22}}x_{33} \\
    \frac{\partial E}{\partial w_{12}}&=\frac{\partial E}{\partial y_{11}}x_{12}+\frac{\partial E}{\partial y_{12}}x_{14}+\frac{\partial E}{\partial y_{21}}x_{32}+\frac{\partial E}{\partial y_{22}}x_{34} \\
    \frac{\partial E}{\partial w_{13}}&=\frac{\partial E}{\partial y_{11}}x_{13}+\frac{\partial E}{\partial y_{12}}x_{15}+\frac{\partial E}{\partial y_{21}}x_{33}+\frac{\partial E}{\partial y_{22}}x_{35} \\
    \frac{\partial E}{\partial w_{21}}&=\frac{\partial E}{\partial y_{11}}x_{21}+\frac{\partial E}{\partial y_{12}}x_{23}+\frac{\partial E}{\partial y_{21}}x_{41}+\frac{\partial E}{\partial y_{22}}x_{43} \\
    \frac{\partial E}{\partial w_{22}}&=\frac{\partial E}{\partial y_{11}}x_{22}+\frac{\partial E}{\partial y_{12}}x_{24}+\frac{\partial E}{\partial y_{21}}x_{42}+\frac{\partial E}{\partial y_{22}}x_{44} \\
    \frac{\partial E}{\partial w_{23}}&=\frac{\partial E}{\partial y_{11}}x_{23}+\frac{\partial E}{\partial y_{12}}x_{25}+\frac{\partial E}{\partial y_{21}}x_{43}+\frac{\partial E}{\partial y_{22}}x_{45} \\
    \frac{\partial E}{\partial w_{31}}&=\frac{\partial E}{\partial y_{11}}x_{31}+\frac{\partial E}{\partial y_{12}}x_{33}+\frac{\partial E}{\partial y_{21}}x_{51}+\frac{\partial E}{\partial y_{22}}x_{53} \\
    \frac{\partial E}{\partial w_{32}}&=\frac{\partial E}{\partial y_{11}}x_{32}+\frac{\partial E}{\partial y_{12}}x_{34}+\frac{\partial E}{\partial y_{21}}x_{52}+\frac{\partial E}{\partial y_{22}}x_{54} \\
    \frac{\partial E}{\partial w_{33}}&=\frac{\partial E}{\partial y_{11}}x_{33}+\frac{\partial E}{\partial y_{12}}x_{35}+\frac{\partial E}{\partial y_{21}}x_{53}+\frac{\partial E}{\partial y_{22}}x_{55}
    \end{aligned}
    $$

    > NOTA: tener en cuenta que en el cálculo del backward, siempre el stride de la full-convolution se establece a 1, independientemente del stride que se utilizó en el forward. Esto se debe a que en la propagación hacia atrás se calcula el gradiente de la función de error con respecto a cada elemento de la entrada y el kernel, por lo que es necesario considerar cada posición del filtro específico sobre cada píxel de la entrada correspondiente.
    """)
    return


@app.cell
def _(layer, np, y):
    # Define el gradiente de la salida
    g = np.random.rand(*y.shape)

    # Propaga el gradiente hacia atrás a través de la convolución
    layer_grad = layer.backward(g)
    layer_grad
    return g, layer_grad


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comprobaciones con PyTorch
    """)
    return


@app.cell
def _(padding, stride, w, x):
    from edunn.utils import reference
    y_torch = reference.conv2d_forward(x, w, stride=stride, padding=padding)
    conv = None
    tnn = None
    torch = None
    x_1 = None
    return conv, tnn, torch, x_1, y_torch
@app.cell
def _(utils, y, y_torch):
    utils.check_same(y_torch,y)
    return


@app.cell
def _(conv, g, torch, x_1, y_torch):
    
    return
@app.cell
def _(layer_grad, utils, x_1):
    
    return
@app.cell
def _(conv, layer_grad, utils):
    
    return
@app.cell
def _(nn, utils):
    samples = 100
    batch_size = 2
    din = 3  # dimensión de entrada
    dout = 5  # dimensión de salida
    input_shape = (batch_size, din, 5, 5)
    layer_1 = nn.Conv2d(input_shape[1], dout, kernel_size=(3, 3))
    # Verificar las derivadas de un modelo de Convolución
    # con valores aleatorios de `w`, `b`, y `x`, la entrada
    utils.check_gradient.common_layer(layer_1, input_shape, samples=samples, tolerance=1e-05)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comprobaciones para 1D
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PyTorch
    """)
    return


@app.cell
def _():
    stride_1, padding_1 = (1, 0)
    return padding_1, stride_1


@app.cell
def _(torch):
    
    x1d = None
    x2d = None
    return x1d, x2d
@app.cell
def _(padding_1, stride_1, tnn, x1d):
    
    conv1d = None
    conv2d = None
    return conv1d, conv2d
@app.cell
def _(conv1d, conv2d, x1d, x2d):
    
    output1d = None
    output2d = None
    return output1d, output2d
@app.cell
def _(output2d):
    # (batch_size, num_filters, 1, new_length) -> (batch_size, num_filters, new_length)
    output2d_1 = output2d.squeeze(2)
    return (output2d_1,)


@app.cell
def _(output1d, output2d_1, torch):
    
    return
@app.cell
def _(output1d, output2d_1, torch):
    
    grad_output_np = None
    return (grad_output_np,)
@app.cell
def _(torch, x1d, x2d):
    
    return
@app.cell
def _(conv1d, conv2d, torch):
    
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### edunn
    """)
    return


@app.cell
def _(conv2d, nn, padding_1, stride_1, x2d):
    x_2 = x2d.detach().numpy()
    w_2 = conv2d.weight.data.detach().numpy()
    _kernel_initializer = nn.initializers.Constant(w_2)
    layer_2 = nn.Conv2d(x_2.shape[1], w_2.shape[0], kernel_size=w_2.shape[2:], stride=stride_1, padding=padding_1, kernel_initializer=_kernel_initializer)
    return layer_2, x_2


@app.cell
def _(layer_2, x_2):
    y_1 = layer_2.forward(x_2)
    (y_1, y_1.shape)
    return (y_1,)


@app.cell
def _(output2d_1, utils, y_1):
    utils.check_same(output2d_1.detach().numpy(), y_1.squeeze(2), tol=1e-05)
    return


@app.cell
def _(grad_output_np, layer_2):
    g_2 = grad_output_np
    layer_grad_1 = layer_2.backward(g_2)
    layer_grad_1
    return (layer_grad_1,)


@app.cell
def _(layer_grad_1, utils, x2d):
    
    return
@app.cell
def _(conv2d, layer_grad_1, utils):
    utils.check_same(conv2d.weight.grad.squeeze(2).detach().numpy(), layer_grad_1[1]['w'].squeeze(2), tol=1e-05)
    return


if __name__ == "__main__":
    app.run()
