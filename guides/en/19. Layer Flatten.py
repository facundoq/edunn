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
    # Capa Flatten

    La operación de aplanamiento (Flatten) en una red neuronal convolucional es bastante sencilla. En el método `forward`, simplemente necesitas cambiar la forma del tensor de entrada a un vector, respetando la dimensión de lotes. En el método `backward`, necesitas cambiar la forma del vector de nuevo a la forma original del tensor de entrada. Implementa ambos métodos.
    """)
    return


@app.cell
def _(nn, np):
    np.random.seed(123)

    x = np.random.rand(2,3,5,5)

    layer=nn.Flatten()
    y=layer.forward(x)
    return layer, x, y


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
def _(x):
    from edunn.utils import reference
    y_torch = reference.flatten_forward(x)
    torch = None
    x_1 = None
    return torch, x_1, y_torch
@app.cell
def _(utils, y, y_torch):
    utils.check_same(y_torch,y)
    return


@app.cell
def _(g, torch, x_1, y_torch):
    
    return
@app.cell
def _(layer_grad, utils, x_1):
    
    return
@app.cell
def _(nn, utils):
    samples = 100
    batch_size = 2
    din = 3  # dimensión de entrada
    dout = 5  # dimensión de salida
    input_shape = (batch_size, din, 3, 3)
    layer_1 = nn.Flatten()
    # Verificar las derivadas de un modelo de Flatten
    # con valores aleatorios de `w`, `b`, y `x`, la entrada
    utils.check_gradient.common_layer(layer_1, input_shape, samples=samples, tolerance=1e-05)
    return


if __name__ == "__main__":
    app.run()
