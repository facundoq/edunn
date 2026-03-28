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
    # Error medio o promedio

    Si bien la capa SquaredError nos permite calcular los errores de cada ejemplo, para obtener una medida del error respecto a un lote o conjunto de ejemplos, tenemos que calcular el promedio de estos errores. Como este cálculo es independiente de la función de error, podemos encapsularlo en su propia clase. Implementar el método `forward` de la clase error medio.

    Nota: Muchas veces a la función de error también se la llama _loss_, para distinguirla del error promedio
    """)
    return


@app.cell
def _(nn, np, utils):
    y = np.array([[2,-2],
                 [-4,4]])
    y_true = np.array([[3,3],
                 [-5,2]])


    layer=nn.MeanError(nn.SquaredError())
    E=15.5
    utils.check_same_float(E,layer.forward(y,y_true),title="mean error")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backward

    Para avanzar más rápido, y porque contiene algún truquillo, el paso `backward` ya está implementado, pero te sugerimos pensar como lo implementarías y luego compararlo la implementación de referencia.
    """)
    return


if __name__ == "__main__":
    app.run()
