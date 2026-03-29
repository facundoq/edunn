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
    # Capa Batch Normalization

    La normalización por lotes aborda el problema de la inicialización deficiente de las redes neuronales. Se puede interpretar como hacer un preprocesamiento en cada capa de la red. Obliga a las activaciones en una red a adoptar una distribución gaussiana unitaria al comienzo del entrenamiento. Esto asegura que todas las neuronas tengan aproximadamente la misma distribución de salida en la red y mejora la tasa de convergencia.

    Explicar el por qué la distribución de las activaciones en una red importa excede el propósito de estas guías, pero de ser de tu interés podés referirte a las páginas 46 — 62 en las [diapositivas de la conferencia](http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture07.pdf) ofrecidas por el curso de la universidad de Stanford.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Método Forward

    Digamos que tenemos un lote de activaciones $x$ en una capa, la versión de $x$ con media cero y varianza unitaria $\hat{x}$ es:

    $$\hat{x}^{(k)}=\frac{x^{(k)}-\mathbb{E}[x^{(k)}]}{\sqrt{\text{Var}[x^{(k)}]}}$$

    Esta es en realidad una operación diferenciable, por eso podemos aplicar la normalización por lotes en el entrenamiento.

    El cálculo de ésta se resume en computar la media $\mu_\mathcal{B}$ y varianza $\sigma_\mathcal{B}^2$ de un mini-batch de $\mathcal{B}=\{x_1, \dots, x_N\}$. Los parámetros aprendibles de la capa son $\gamma$ y $\beta$ que son utilizados para escalar y desplazar los valores normalizados.

    $$
    \begin{aligned}
    \mu_\mathcal{B} &= \frac{1}{N} \sum_{i=1}^{N} x_i & \text{(mini-batch mean)} \\
    \sigma_\mathcal{B}^2 &= \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu_\mathcal{B})^2 & \text{(mini-batch variance)} \\
    \hat{x}_i &= \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}} & \text{(normalize)} \\
    \text{\textbf{BN}}_{\gamma,\beta}(x_i) &\stackrel{\text{def}}{=} \gamma \hat{x}_i + \beta = y_i & \text{(scale and shift)}
    \end{aligned}
    $$

    > NOTA: En la implementación, insertamos la capa `BatchNorm` justo después de una capa `Dense` o una capa `Conv2d`, y antes de las capas no lineales.
    """)
    return


@app.cell
def _(nn, np):
    np.random.seed(123)
    din = 10
    _batch_size = 2
    x = np.random.rand(_batch_size, din)
    w = np.random.rand(din)
    b = np.random.rand(din)
    gamma_initializer = nn.initializers.Constant(w)
    beta_initializer = nn.initializers.Constant(b)
    layer = nn.BatchNorm(num_features=din, gamma_initializer=gamma_initializer, beta_initializer=beta_initializer)
    return b, layer, w, x


@app.cell
def _(x):
    x
    return


@app.cell
def _(layer, x):
    y = layer.forward(x)
    y, y.shape
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Método Backward

    ### `δEδβ`

    El cálculo de los gradientes del error $E$ con respecto al parámetro $\beta$ se puede hacer derivando parcialmente como se explicó en guías anteriores:

    $$
    \frac{\partial E}{\partial \beta} = \frac{\partial E}{\partial y} \cdot \frac{\partial y}{\partial \beta}
    $$

    Como $y$ es un vector de $N$ elementos, tenemos que sumar por todos sus valores para aplicar la regla de la cadena:

    $$
    \frac{\partial E}{\partial \beta} = \frac{\partial E}{\partial y_1} \cdot \frac{\partial y_1}{\partial \beta} + \cdots + \frac{\partial E}{\partial y_N} \cdot \frac{\partial y_N}{\partial \beta}
    \qquad \text{donde} \qquad
    \frac{\partial y_i}{\partial \beta} = \frac{\partial (\gamma \hat{x}_i + \beta)}{\partial \beta} = 1
    $$

    de este modo:

    $$
    \frac{\partial E}{\partial \beta} = \sum\limits_{i=1}^N \frac{\partial E}{\partial y_i} \cdot 1
    $$

    ### `δEδγ`

    El cálculo de los gradientes del error $E$ con respecto al parámetro $\gamma$ se puede hacer derivando parcialmente como se explicó en guías anteriores:

    $$
    \frac{\partial E}{\partial \gamma} = \frac{\partial E}{\partial y} \cdot \frac{\partial y}{\partial \gamma}
    $$

    Como $y$ es un vector de $N$ elementos, tenemos que sumar por todos sus valores para aplicar la regla de la cadena:

    $$
    \frac{\partial E}{\partial \gamma} = \frac{\partial E}{\partial y_1} \cdot \frac{\partial y_1}{\partial \gamma} + \cdots + \frac{\partial E}{\partial y_N} \cdot \frac{\partial y_N}{\partial \gamma}
    \qquad \text{donde} \qquad
    \frac{\partial y_i}{\partial \gamma} = \frac{\partial (\gamma \hat{x}_i + \beta)}{\partial \gamma} = \hat{x}_i
    $$

    de este modo:

    $$
    \frac{\partial E}{\partial \gamma} = \sum\limits_{i=1}^N \frac{\partial E}{\partial y_i} \cdot \hat{x}_i
    $$

    ### `δEδx`

    <!-- Utilizando la regla de la cadena para el cálculo diferencial, esta nos dice que la derivada de una función compuesta es el producto de las derivadas de las funciones que la componen. -->

    Teniendo en cuenta de qué depende cada función:

    <center>

    ||||
    |:-:|:-:|:-:|
    |$E(y)$|$y(\hat{x},\gamma,\beta)$|$\hat{x}(\mu,\sigma^2,x)$|

    </center>

    Obtenemos que:

    $$
    \dfrac{\partial E}{\partial x_i} = \frac{\partial E}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial E}{\partial \mu} \cdot \frac{\partial \mu}{\partial x_i} + \frac{\partial E}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial x_i}
    $$

    En las siguientes subsecciones calcularemos la expresión correspondiente para el gradiente de cada componente.

    #### `δEδx̂`

    $$
    \frac{\partial E}{\partial \hat{x}_i} = \frac{\partial E}{\partial y_i} \cdot \frac{\partial y_i}{\partial \hat{x}_i} = \frac{\partial E}{\partial y_i} \cdot \frac{\partial (\gamma \hat{x}_i + \beta)}{\partial \hat{x}_i} = \frac{\partial E}{\partial y_i} \cdot \gamma
    $$

    #### `δEδμ`

    Notar que $\sigma^2$ se puede escribir en función de $\mu$, es por ello que $E$ depende de $\mu$ a traves de dos variables: $\hat{x}_i$​ y $\sigma^2$.

    $$
    \dfrac{\partial E}{\partial \mu} = \frac{\partial E}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial \mu} + \frac{\partial E}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial\mu}
    $$

    Calculando las derivadas parciales para `δx̂δμ` y `δσ²δμ`:

    $$
    \begin{aligned}
    \hat{x}_i = \frac{(x_i - \mu)}{\sqrt{\sigma^2 + \epsilon}}
    &\qquad \Rightarrow \qquad
    \dfrac{\partial \hat{x}_i}{\partial \mu} = \frac{-1}{\sqrt{\sigma^2 + \epsilon}} \\
    \sigma^2 = \frac{1}{N} \sum\limits_{i=1}^N (x_i - \mu)^2
    &\qquad \Rightarrow \qquad
    \dfrac{\partial \sigma^2}{\partial \mu} = \frac{1}{N} \sum\limits_{i=1}^N -2 \cdot (x_i - \mu) \\
    \end{aligned}
    $$

    Reemplazando éstas últimas y dejando como variables a los gradientes del error $E$, obtenemos:

    $$
    \begin{aligned}
    \frac{\partial E}{\partial \mu} &= \bigg(\sum\limits_{i=1}^N  \frac{\partial E}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}} \bigg) + \bigg( \frac{\partial E}{\partial \sigma^2} \cdot \frac{1}{N} \sum\limits_{i=1}^N -2(x_i - \mu)   \bigg) \qquad \\
    &= \bigg(\sum\limits_{i=1}^N  \frac{\partial E}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}} \bigg) + \bigg( \frac{\partial E}{\partial \sigma^2} \cdot (-2) \cdot \bigg( \frac{1}{N} \sum\limits_{i=1}^N x_i - \frac{1}{N} \sum\limits_{i=1}^N \mu   \bigg) \bigg) \qquad \\
    &= \bigg(\sum\limits_{i=1}^N  \frac{\partial E}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}} \bigg) + \bigg( \frac{\partial E}{\partial \sigma^2} \cdot (-2) \cdot \underbrace{\bigg( \mu - \frac{N \cdot \mu}{N} \bigg)}_{0} \bigg) \qquad \\
    &= \sum\limits_{i=1}^N  \frac{\partial E}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}} \qquad \\
    \end{aligned}
    $$

    #### `δEδσ²`

    $$\frac{\partial E}{\partial \sigma^2} = \frac{\partial E}{\partial \hat{x}} \cdot \frac{\partial \hat{x}}{\partial \sigma^2}$$

    Reescribiendo $\hat{x}_i$ para que su derivada sea más fácil de calcular, vemos que $(x_i - \mu)$ pasa a ser un factor constante, de modo que:

    $$
    \hat{x}_i = (x_i - \mu)(\sigma^2 + \epsilon)^{-0.5}
    \qquad \Rightarrow \qquad
    \dfrac{\partial \hat{x}}{\partial \sigma^2} = -0.5 \sum\limits_{i=1}^N (x_i - \mu) \cdot (\sigma^2 + \epsilon)^{-1.5}
    $$

    #### `δEδx` (cont.)

    Calculando las derivadas parciales restantes (`δx̂δx`, `δμδx` y `δσ²δx`) de la expresión original obtenemos que:

    <center>

    ||||
    |:-:|:-:|:-:|
    |$\dfrac{\partial \hat{x}_i}{\partial x_i} = \dfrac{1}{\sqrt{\sigma^2 + \epsilon}}$|$\dfrac{\partial \mu}{\partial x_i} = \dfrac{1}{N}$|$\dfrac{\partial \sigma^2}{\partial x_i} = \dfrac{2(x_i - \mu)}{N}$|

    </center>

    Finalmente podemos calcular el gradiente del error $E$ con respecto a $x$ utilizando el siguiente truco:

    $$
    (\sigma^2 + \epsilon)^{-1.5} = (\sigma^2 + \epsilon)^{-0.5}(\sigma^2 + \epsilon)^{-1} = (\sigma^2 + \epsilon)^{-0.5} \frac{1}{\sqrt{\sigma^2 + \epsilon}}\frac{1}{\sqrt{\sigma^2 + \epsilon}}
    $$

    de este modo:

    $$
    \begin{aligned}
    \frac{\partial E}{\partial x_i} &= \bigg(\frac{\partial E}{\partial \hat{x}_i} \cdot \dfrac{1}{\sqrt{\sigma^2 + \epsilon}} \quad\; \bigg) + \bigg(\frac{\partial E}{\partial \mu} \cdot \dfrac{1}{N} \qquad\qquad\qquad\!\! \bigg) + \bigg(\frac{\partial E}{\partial \sigma^2} \cdot \dfrac{2(x_i - \mu)}{N}\bigg) \qquad \\
    &= \bigg(\frac{\partial E}{\partial \hat{x}_i} \cdot \dfrac{1}{\sqrt{\sigma^2 + \epsilon}} \quad\; \bigg) + \bigg(\frac{1}{N} \sum\limits_{j=1}^N  \frac{\partial E}{\partial \hat{x}_j} \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}}\bigg) + \bigg(-0.5 \sum\limits_{j=1}^N \frac{\partial E}{\partial \hat{x}_j} \cdot (x_j - \mu) \cdot (\sigma^2 + \epsilon)^{-1.5} \cdot \dfrac{2(x_i - \mu)}{N} \bigg) \qquad \\
    &= \bigg(\frac{\partial E}{\partial \hat{x}_i} \cdot (\sigma^2 + \epsilon)^{-0.5} \bigg) - \bigg(\frac{(\sigma^2 + \epsilon)^{-0.5}}{N} \sum\limits_{j=1}^N  \frac{\partial E}{\partial \hat{x}_j} \;\, \bigg) - \bigg(\frac{(\sigma^2 + \epsilon)^{-0.5}}{N} \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \sum\limits_{j=1}^N \frac{\partial E}{\partial \hat{x}_j} \cdot \frac{(x_j - \mu)}{\sqrt{\sigma^2 + \epsilon}} \bigg )\qquad \\
    &= \bigg(\frac{\partial E}{\partial \hat{x}_i} \cdot (\sigma^2 + \epsilon)^{-0.5} \bigg) - \bigg(\frac{(\sigma^2 + \epsilon)^{-0.5}}{N} \sum\limits_{j=1}^N  \frac{\partial E}{\partial \hat{x}_j} \;\, \bigg) - \bigg(\frac{(\sigma^2 + \epsilon)^{-0.5}}{N} \cdot \hat{x}_i \sum\limits_{j=1}^N \frac{\partial E}{\partial \hat{x}_j} \cdot \hat{x}_j \bigg )\qquad \\
    &= \boxed{\frac{(\sigma^2 + \epsilon)^{-0.5}}{N} \bigg [N \frac{\partial E}{\partial \hat{x}_i} - \sum\limits_{j=1}^N  \frac{\partial E}{\partial \hat{x}_j} - \hat{x}_i \sum\limits_{j=1}^N \frac{\partial E}{\partial \hat{x}_j} \cdot \hat{x}_j\bigg ]} \qquad \\
    &= \frac{(\sigma^2 + \epsilon)^{-0.5}}{N} \bigg [N \frac{\partial E}{\partial y_i} \cdot \gamma - \sum\limits_{j=1}^N  \frac{\partial E}{\partial y_j} \cdot \gamma - \hat{x}_i \sum\limits_{j=1}^N \frac{\partial E}{\partial y_j} \cdot \gamma \cdot \hat{x}_j\bigg ] \qquad \\
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(layer, np, y):
    # Define el gradiente de la salida
    g = np.random.rand(*y.shape)

    # Propaga el gradiente hacia atrás a través de la convolución
    layer_grad = layer.backward(g)
    layer_grad
    return (layer_grad,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comprobaciones con PyTorch
    """)
    return


@app.cell
def _(b, w, x):
    from edunn.utils import reference
    y_torch = reference.batchnorm_forward(x, w, b)
    batch_norm = None
    torch = None
    x_1 = None
    return batch_norm, y_torch


@app.cell
def _(utils, y, y_torch):
    utils.check_same(y_torch,y,tol=1e-5)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(batch_norm, layer_grad, utils):
    utils.check_same(batch_norm.weight.grad.numpy(),layer_grad[1]['w'],tol=1e-5)
    return


@app.cell
def _(batch_norm, layer_grad, utils):
    utils.check_same(batch_norm.bias.grad.numpy(),layer_grad[1]['b'],tol=1e-5)
    return


@app.cell
def _(nn, utils):
    samples = 100
    _batch_size = 2
    din_1 = 10
    input_shape = (_batch_size, din_1)
    layer_1 = nn.BatchNorm(din_1)
    utils.check_gradient.common_layer(layer_1, input_shape, samples=samples, tolerance=1e-05)
    return


if __name__ == "__main__":
    app.run()
