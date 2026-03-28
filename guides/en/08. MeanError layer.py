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
    # Mean Error

    The SquaredError layer allows us to calculate the errors for each example.

    However, to obtain a measure of error for a batch or set of examples, we need to calculate the average of these errors.

    Since this calculation is independent of the error function, we can encapsulate it in its own class. Implement the `forward` method of the MeanError class.

    Note: The error function is often referred to as 'loss' to distinguish it from the mean error.
    """)
    return


@app.cell
def _(nn, np, utils):
    y = np.array([[2,-2],
                 [-4,4]])
    y_true = np.array([[3,3],
                 [-5,2]])


    layer = nn.MeanError(nn.SquaredError())
    E = 15.5
    utils.check_same_float(E, layer.forward(y, y_true), title="mean error")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backward

    To make progress faster, and because it contains some tricks, the `backward` step is already implemented, but we suggest thinking about how you would implement it and then comparing it to the reference implementation.
    """)
    return


if __name__ == "__main__":
    app.run()
