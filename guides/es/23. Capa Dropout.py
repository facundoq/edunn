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

    return nn, np


@app.cell
def _(np, sys):
    np.set_printoptions(threshold=sys.maxsize)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Capa Dropout

    Dropout actúa como una técnica de _regularización_ que elimina o desactiva nodos en la propagación hacia adelante, lo que hace que la red sea menos propensa al sobreajuste al evitar que la red dependa demasiado de cualquier neurona individual. Esta capa no tiene parámetros.

    * En la propagación hacia adelante, las entradas se establecen en cero con una probabilidad $p$, y de lo contrario se escalan por $\frac{1}{1-p}$.

      - La propagación hacia adelante durante el entrenamiento solo se utiliza para configurar la red para la propagación hacia atrás, donde la red se modifica realmente.

      - Para cada neurona individual en la capa, podemos decir que $x \sim B(1, p)$, ya que estamos considerando un solo "experimento" (la activación o desactivación de la neurona) con una probabilidad de éxito de $p$.

    * En la propagación hacia atrás, los gradientes para las mismas unidades eliminadas se anulan; otros gradientes se escalan por el mismo factor de $\frac{1}{1-p}$.

      - Es decir, si un nodo fue eliminado por la capa Dropout, entonces su influencia (el gradiente) en los pesos salientes es también 0 (ya que $0 * w_i = 0$). En resumen, la propagación hacia atrás funciona como siempre.

    > NOTA: tener en cuenta que durante la fase de prueba o validación, todas las neuronas están activas (es decir, no se aplica Dropout) para obtener una predicción basada en toda la red.
    """)
    return


@app.cell
def _(nn, np):
    np.random.seed(123)

    din=10
    batch_size=2

    x = np.random.rand(batch_size,din)
                       
    layer=nn.Dropout(p=0.5)
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
def _(layer, np, y):
    # Define el gradiente de la salida
    g = np.random.rand(*y.shape)

    # Propaga el gradiente hacia atrás a través de la convolución
    layer_grad = layer.backward(g)
    layer_grad
    return (g,)


@app.cell
def _(layer, x):
    from edunn.model import Phase
    layer.set_phase(Phase.Test)
    y_1 = layer.forward(x)
    (y_1, y_1.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comprobaciones con PyTorch
    """)
    return


@app.cell
def _(x):
    from edunn.utils import reference
    y_torch, mask = reference.dropout_forward(x, p=0.5)
    dropout = None
    torch = None
    x_1 = None
    return dropout, torch, x_1, y_torch
@app.cell
def _(g, torch, x_1, y_torch):
    
    return
@app.cell
def _(dropout, x_1):
    dropout.eval()
    y_torch_1 = dropout(x_1)
    y_torch_1
    return


if __name__ == "__main__":
    app.run()
