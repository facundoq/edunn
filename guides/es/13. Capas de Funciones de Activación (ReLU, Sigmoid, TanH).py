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
    # Funciones de activación

    Una `red neuronal` es un modelo que combina  transformaciones de varias capas como las de Regresión Lineal o Regresión Logística. Pero para definir una red, debemos utilizar _funciones de activación_, que permiten hacer transformaciones no lineales de los datos. De otra forma, la combinación de dos capas de Regresión Lineal, por ejemplo, es equivalente a tener una sola, ya que solo va a poder representar una recta. En ese caso, nuestra red no va a ser más potente, y solo será más ineficiente.

    Hay distintos tipos de funciones de activación: vamos a ver tres de las más comunes, `relu`, `sigmoid` (o `logística`) y `tanh` (tangente hiperbólica). Cada una tiene distintas propiedades:

    * `relu`: es eficiente para calcular tanto su salida como su derivada, y aunque su no-linealidad es simple sirve para darle mayor poder de aproximación a una red. Por otro lado, su salida está en el rango $[0,\infty)$, con lo cual no sirve para algunas tareas
    * `sigmoid`: es eficiente, aunque no tanto como `relu`, y como su rango es $(0,1)$ permite codificar un valor que puede interpretarse como una probabilidad
    * `tanh`: Es similar a `sigmoid`, pero su rango es $(-1,1)$, con lo cual permite codificar valor como positivos y negativos y así generar representaciones con feedback de esos signos en la red.

    <div style="background-color: white;">
    <img src="img/relu.png" width="25%" style="display:inline-block"> <img src="img/sigmoid.png" width="25%" style="display:inline-block"> <img src="img/tanh.png" width="25%" style="display:inline-block">
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Capa ReLU

    La función `ReLU` (Rectified Linear Unit) es extremadamente simple.

    <div style="background-color: white;">
    <img src="img/relu.png" width=50%>
    </div>

    Formalmente,

    $$ReLU(x) = \begin{cases}
        0 & \text{if } x \le 0 \\
        1 & \text{if }  x > 0
    \end{cases}
    $$

    De forma más corta pero un poco más difícil de entender, podemos escribir `ReLU` como:
    $$ReLU(x) = max(0,x)$$

    En términos de código, calcular `relu` para un solo valor es simple:

    ````python
    def relu(x:float):
        if x>0:
            return x
        else:
            return 0
    ````

    No obstante, en este caso tené en cuenta que recibirás un tensor de valores. De todas formas `ReLU`, como las otras funciones de activación, es fácil de cálcular ya que se aplica _elemento a elemento_. Es decir, `ReLU` de un vector es equivalente a aplicar `ReLU` a cada uno de sus valores, o sea:

    $$ReLU( (-2,4,7)=( ReLU(-2), ReLU(4), ReLU(7)) = (0,4,7)$$

    Para el caso de una matriz o un tensor en general, se procede de igual forma.

    Implementá el método `forward` de la clase `ReLU` en el archivo `edunn/models/activations.py`
    """)
    return


@app.cell
def _(nn, np):
    _x = np.array([[3.5, -7.2, 5.3], [-3.5, 7.2, -5.3]])
    _layer = nn.ReLU()
    _y = np.array([[3.5, 0, 5.3], [0, 7.2, 0]])
    nn.utils.check_same(_y, _layer.forward(_x))
    # plot values
    nn.plot.plot_activation_function(_layer)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método backward de `ReLU`

    Afortunadamente, el método `backward` de `ReLU` es sencillo. Procediendo por casos, podemos ver que si el forward es :

    $$ReLU(x) = \begin{cases}
        0 & \text{if } x \le 0 \\
        x & \text{if }  x > 0
    \end{cases}
    $$

    Podemos derivar cada caso por separado, y entonces:
    $$\frac{d ReLU(x)}{dx} = \begin{cases}
        0 & \text{if } x \le 0 \\
        1 & \text{if }  x > 0
    \end{cases}
    $$

    En el caso de $x=0$, en verdad la derivada de $ReLU$ no está definida; no obstante, en este caso podemos redefinir dicha derivada como 0 (o 1), y no afectará realmente a la optimización.
    """)
    return


@app.cell
def _(nn):
    from edunn.utils import check_gradient
    _samples = 100
    _input_shape = (5, 2)
    # Cantidad de ejemplos aleatorios y tamaño de los mismo gpara generar 
    # muestras de x y verificar las derivadas
    _layer = nn.ReLU()
    check_gradient.common_layer(_layer, _input_shape, samples=_samples)
    # Verificar derivadas de una función ReLU
    nn.plot.plot_activation_function(nn.ReLU(), backward=True)
    return (check_gradient,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Capa Sigmoid

    La función `sigmoid` (sigmoidea o también conocida como logística) convierte cualquier valor al intervalo $(0,1)$.

    <div style="background-color: white;">
    <img src="img/sigmoid.png" width=50%>
    </div>

    Su definición es:

    $$
    Sigmoid(x)= \frac{1}{1+e^{-x}}
    $$

    De esta forma, por ejemplo, tenemos:

    $$ sigmoid(0)=\frac{1}{1+1}=\frac{1}{2}=0.5$$
    $$ sigmoid(1)=\frac{1}{1+e^{-1}}=\frac{1}{1+0.36}=0.73$$
    $$ sigmoid(-1)=\frac{1}{1+e^{-(-1)}}=\frac{1}{1+2.71}=0.26$$

    Como vemos entonces, el balance de la función está en el valor $0$, para el cual la salida es $0.5$; valores mayores a $0$ causan salidas mayores, y viceversa.

    Implementá el método `forward` de la clase `Sigmoid` en el archivo `edunn/models/activations.py`
    """)
    return


@app.cell
def _(nn, np):
    _x = np.array([[0, 1, -1]])
    _layer = nn.Sigmoid()
    _y = np.array([[0.5, 0.73105858, 0.26894142]])
    nn.utils.check_same(_y, _layer.forward(_x), tol=1e-06)
    nn.plot.plot_activation_function(_layer)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método backward de `Sigmoid`

    Las derivadas del método `backward` de `Sigmoid` son sencillas, pero además vamos a querer llevarlas a una forma simplificada que hace más eficiente su cálculo.

    $$
    Sigmoid(x)= \frac{1}{1+e^{-x}}
    $$

    $$
    \begin{aligned}
    \frac{d Sigmoid(x)}{dx}
    &= \frac{d \frac{1}{1+e^{-x}}}{dx} =\frac{d (1+e^{-x})^{-1}}{dx} \\
    &= \frac{d (1+e^{-x})^{-1}}{d(1+e^{-x})} \frac{d (1+e^{-x})}{dx} & \text{(regla de la cadena con $g(x)=1+e^{-x}$)} \\
    &= \frac{d  (1+e^{-x})^{-1} }{d(1+e^{-x})} (-e^{-x}) & \text{(derivada de $1+e^{-x}$)} \\
    &=  -(1+e^{-x})^{-2} (-e^{-x}) & \text{(derivada de $g(x)^{-1}=-g(x)^{-2}$)} \\
    &=  (1+e^{-x})^{-2} e^{-x}\\
    &=  Sigmoid(x)^2 e^{-x}\\
    \end{aligned}
    $$

    En este punto, tenemos una fórmula para la derivada de $Sigmoid$, pero como decíamos antes, vamos a llevarla a una fórmula más simple que solo utilice el valor original de $Sigmoid$. De esta forma, nos evitaremos volver a realizar operaciones de exponenciación o división que son más costosas computacionalmente. Para eso, comenzamos con la anteúltima línea de la derivación anterior:

    <!-- $$
    \begin{aligned}
    \frac{d Sigmoid(x)}{dx} &=  (1+e^{-x})^{-2} e^{-x}\\
    &=  Sigmoid(x)^2 e^{-x}\\
    &=  Sigmoid(x)^2 (1-1+e^{-x})\\
    &=  Sigmoid(x)^2 (1-Sigmoid(x)^{-1})\\
    &=  Sigmoid(x) Sigmoid(x) (1-Sigmoid(x)^{-1})\\
    &=  Sigmoid(x) [Sigmoid(x) (1-Sigmoid(x)^{-1})]\\
    &=  Sigmoid(x) [Sigmoid(x) * 1- Sigmoid(x) Sigmoid(x)^{-1}]\\
    &=  Sigmoid(x) (Sigmoid(x) -1)\\
    \end{aligned}
    $$ -->

    <!-- https://www.desmos.com/calculator/ekqs9htkml -->

    $$
    \begin{aligned}
    \frac{d Sigmoid(x)}{dx} &=  (1+e^{-x})^{-2} e^{-x}\\
    &= \frac{e^{-x}}{(1 + e^{-x})^2} \\
    &= \frac{1}{1 + e^{-x}} \frac{e^{-x}}{1 + e^{-x}} \\
    &= \frac{1}{1 + e^{-x}} \frac{(1 + e^{-x}) - 1}{1 + e^{-x}} \\
    &= \frac{1}{1 + e^{-x}} \left(1 - \frac{1}{1 + e^{-x}}\right) \\
    &= Sigmoid(x) (1 - Sigmoid(x))
    \end{aligned}
    $$

    Como podemos ver, esta fórmula final $Sigmoid'(x) = Sigmoid(x)  (1-Sigmoid(x))$ nos dice que si almacenamos el valor de $Sigmoid(x)$ en el paso `forward`, entonces para el `backward` la derivada se calcula simplemente como `y*(1-y)` que solo requiere una suma y una multiplicación de vectores.

    Implementá el método `backward` de la clase `Sigmoid` y verificalo con el siguiente código:
    """)
    return


@app.cell
def _(check_gradient, nn):
    _samples = 100
    _input_shape = (5, 2)
    _layer = nn.Sigmoid()
    check_gradient.common_layer(_layer, _input_shape, samples=_samples)
    nn.plot.plot_activation_function(_layer, backward=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Capa `TanH`

    La función `tanh` (tangente hiperbólica) convierte cualquier valor al intervalo $(-1,1)$.

    <div style="background-color: white;">
    <img src="img/tanh.png" width=35%>
    </div>

    $tanh$ es una función geométrica [hiperbólica](https://en.wikipedia.org/wiki/Hyperbolic_functions). Al igual que $tan(x)=\frac{cos(x)}{sen(x)}$, $tanh$ se define a partir del coseno y seno hiperbólicos:

    $$
    tanh(x)= \frac{cosh(x)}{senh(x)} = \frac{e^x-e^{-x}}{e^x+e^{-x}}
    $$

    Por ejemplo:

    $$
    \begin{aligned}
    tanh(0)  &= \frac{1-1}{1+1} &&= \frac{0}{2} &&= 0 \\
    tanh(1)  &= \frac{e-e^{-1}}{e+e^{-1}} &&= \frac{2.35}{3.08} &&= 0.76 \\
    tanh(-1) &= \frac{e^{-1}-e}{e^{-1}+e} &&= \frac{-2.35}{3.08} &&= -0.76
    \end{aligned}
     $$

    El balance de la función está en el valor $0$, para el cual la salida es $0$; valores mayores a $0$ causan salidas mayores, y viceversa, por eso $tanh$ es una función impar ($tanh(x)=-tanh(-x)$).

    $TanH$ es muy similar a la función $Sigmoid$. De hecho, con el siguiente gráfico podemos observar que $TanH$ es simplemente $Sigmoid$, pero:

    * Multiplicada por dos (para convertir el rango $(0,1)$ al rango $(0,2)$)
    * Restándole 1 (para convertir el rango $(0,2)$ al rango $(-1,1)$)
    * Multiplicando a $x$ por 2, para que la curva de ambas sea igual

    <div style="background-color: white;">
    <img src="img/sigmoid.png" width=35%>
    </div>

    En verdad, [se puede definir](http://facundoq.github.io/edunn/material/sigmoid_tanh) a $tanh$ en base a $Sigmoid$:
    $$
    tanh(x) = sigmoid(2x)*2-1
    $$

    _Fe de Erratas_ del apunte previo:

    $$
    \begin{aligned}
    tanh(x) &= \frac{e^x-e^{-x}}{e^x+e^{-x}} \\
    &= \frac{e^x}{e^x} \frac{e^x-e^{-x}}{e^x+e^{-x}} \\
    &= \frac{e^{2x}-1}{e^{2x}+1} \\
    &= \frac{e^{w}-1}{e^{w}+1} & \text{(sea $w=2x$)} \\
    &= \frac{-1+e^{w}+1-1}{1+e^{w}} \\
    &= \frac{-2+e^{w}+1}{1+e^{w}} \\
    &= \frac{-2}{1+e^{w}} +  \frac{e^{w}+1}{1+e^{w}} \\
    &= -2 Sigmoid(-w) + 1 \\
    &= 2 Sigmoid(w) - 1 & \text{(dado que $tanh(x)\stackrel{\text{impar}}{=}-tanh(-x)$)} \\
    &= 2 Sigmoid(2x) - 1 \\
    \end{aligned}
    $$

    Esta forma será mucho más cómoda para la implementación, ya que podemos reutilizar la capa `Sigmoid` tanto para el forward como el backward.

    Implementá el método `forward` de la clase `TanH` en el archivo `edunn/models/activations.py`  _utilizando la capa `Sigmoid`_ (te ayudamos ya definiendo una variable para ello).
    """)
    return


@app.cell
def _(nn, np):
    _x = np.array([[0, 0.5, -0.5]])
    _layer = nn.TanH()
    _y = np.array([[0.0, 0.46211716, -0.46211716]])
    nn.utils.check_same(_y, _layer.forward(_x), tol=1e-06)
    nn.plot.plot_activation_function(_layer)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método backward de `TanH`

    Las derivadas del método `backward` de `TanH` pueden obtenerse en base a las de `Sigmoid`. Dado que

    $$
    tanh(x) = sigmoid(2x)*2-1
    $$

    Entonces:

    $$
    tanh'(x) = (sigmoid'(2x)*2)*2 = sigmoid'(2x)*4
    $$

    Es decir, la derivada de $tanh$ consta simplemente de multiplicar por dos la derivada de $sigmoid$.

    Implementá el método `backward` de la clase `TanH` y verificalo con el siguiente código:
    """)
    return


@app.cell
def _(check_gradient, nn):
    _samples = 100
    _input_shape = (5, 2)
    _layer = nn.TanH()
    check_gradient.common_layer(_layer, _input_shape, samples=_samples)
    nn.plot.plot_activation_function(_layer, backward=True)
    return


if __name__ == "__main__":
    app.run()
