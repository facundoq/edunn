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
    # Gaussian Error Linear Unit

    GELU es una función de activación que se utiliza en las capas ocultas de las redes neuronales. Se utiliza para introducir la no linealidad en los modelos de redes neuronales y ayuda a decidir qué neuronas deben activarse.

    GELU combina las ventajas de otras funciones de activación populares como ReLU y ELU. Proporciona una saturación suave tanto para los valores positivos como para los negativos, lo que puede ayudar a mitigar el problema del desvanecimiento del gradiente.

    ## Método Forward

    Existen distintas formas de expresar tal función, una de ellas es utilizando la función de distribución acumulativa (CDF) y la [función de error](https://en.wikipedia.org/wiki/Error_function):

    $$ \text{GELU} = x \cdot \text{CDF}(x) = x \cdot \frac{1+\text{erf}\big(\frac{1}{\sqrt{2}}\big)}{2} $$

    Esto se debe a que la `CDF` puede ser expresada utilizando la `erf`, donde se escala la función lineal `x` con la `CDF`. Esta versión de GELU es computacionalmente más eficiente y es la que se utiliza en la biblioteca de aprendizaje profundo de PyTorch.

    <center>
    <img src="img/gelu.png" width="25%" style="display:inline-block">
    </center>

    > NOTA: actualmente la función `erf` no es encuentra implementada en `numpy`, puedes tomarte el tiempo de implementarla numéricamente o utilizar la implementación dada por `from scipy.special import erf`.

    ## Método Backward

    Su cálculo es más sencillo de lo que parece, ya que en el mismo no utilizaremos la expresión dependiente de la `erf`, sino la que depende expresamente de `CDF`.

    $$
    \begin{aligned}
    (\text{x} \cdot \text{CDF}(\text{x}))' &= \text{x}' \cdot \text{CDF}(\text{x}) + \text{x} \cdot \text{CDF}'(\text{x}) \\
    &= \text{CDF}(\text{x}) + \text{x} \cdot \text{PDF}(\text{x})
    \end{aligned}
    $$

    Esto se debe al teorema fundamental del cálculo, donde por definición:

    - Derivar la CDF te da la PDF: $\text{PDF}(x) = \frac{d}{dx} \text{CDF}(x)$
    - Integrar la PDF te da la CDF: $\text{CDF}(x) = \int_{-\infty}^{x} \text{PDF}(t) dt$

    La `PDF` que utilizaremos es la de la distribución normal o Gaussiana. Nuevamente, `numpy` no provee su cálculo en relación a los valores de un vector de entrada, por ello deberás implentar la función `normal_pdf` en el archivo `activations.py`.

    Esta es la fórmula para la función de densidad de probabilidad (PDF) de una distribución normal:

    $$
    f(x, \mu=0, \sigma=1) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
    $$

    donde $x$ es la variable aleatoria, $\mu$ es la media de la distribución y $\sigma$ es la desviación estándar de la distribución.

    <!-- La PDF (Función de Densidad de Probabilidad) da la probabilidad de que una variable aleatoria tome un valor específico, mientras que la CDF (Función de Distribución Acumulativa) da la probabilidad de que una variable aleatoria tome un valor menor o igual a un valor dado. -->

    <!-- 1. **PDF (Función de Densidad de Probabilidad)**: Es una función que describe la probabilidad relativa de que una variable aleatoria tome un valor dado. La probabilidad de que la variable aleatoria caiga dentro de un rango particular se dada por el área bajo la gráfica de la función de densidad en ese intervalo.

    2. **CDF (Función de Distribución Acumulativa)**: Es una función que indica la probabilidad de que una variable aleatoria sea menor o igual a un valor dado. Es la integral de la PDF hasta ese valor.

    - La PDF puede tomar valores mayores a 1, mientras que la CDF nunca puede tomar valores mayores a 1. -->
    """)
    return


@app.cell
def _(nn, np):
    np.random.seed(123)

    din=10
    batch_size=2

    x = np.random.rand(batch_size,din)

    layer=nn.GELU()
    return layer, x


@app.cell
def _(x):
    x
    return


@app.cell
def _(layer, x):
    y = layer.forward(x)
    y, y.shape
    return (y,)


@app.cell
def _(layer, nn):
    # plot values
    nn.plot.plot_activation_function(layer.__class__())
    return


@app.cell
def _(layer, np, y):
    # Define el gradiente de la salida
    g = np.random.rand(*y.shape)

    # Propaga el gradiente hacia atrás a través de la convolución
    layer_grad = layer.backward(g)
    layer_grad
    return g, layer_grad


@app.cell
def _(x):
    from edunn.utils import reference
    y_torch = reference.gelu_forward(x)
    torch = None
    x_1 = None
    return torch, x_1, y_torch
@app.cell
def _(utils, y, y_torch):
    utils.check_same(y_torch,y,tol=1e-5)
    return


@app.cell
def _(g, torch, x_1, y_torch):
    
    return
@app.cell
def _(layer_grad, utils, x_1):
    
    return
@app.cell
def _(nn):
    from edunn.utils import check_gradient
    samples = 100
    input_shape = (5, 2)
    # Cantidad de ejemplos aleatorios y tamaño de los mismo gpara generar 
    # muestras de x y verificar las derivadas
    layer_1 = nn.GELU()
    check_gradient.common_layer(layer_1, input_shape, samples=samples)
    # Verificar derivadas de una función GELU
    nn.plot.plot_activation_function(nn.GELU(), backward=True)
    return


if __name__ == "__main__":
    app.run()
