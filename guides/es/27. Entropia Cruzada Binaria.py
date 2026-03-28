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
    # Binary Cross Entropy

    This is a placeholder for the Binary Cross Entropy guide.
    """)
    return


if __name__ == "__main__":
    app.run()
