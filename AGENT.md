# AGENT.md

## Project: edunn

`edunn` (Educational Neural Networks) is a modular neural network framework implemented from scratch using Python and NumPy. It is designed to help learners understand the inner workings of modern deep learning frameworks (like PyTorch or Keras) by re-implementing their core components.

The framework supports defining, training, and using neural networks in a modular fashion, including layers (Linear, Bias, ReLU, Sigmoid, etc.), optimizers (Gradient Descent, SGD), and error functions (Mean Error, Cross-Entropy).

## Project Structure
- `edunn/`: Core library implementation.
- `guides/`: Interactive Jupyter notebooks (English and Spanish) for implementation exercises.
- `animations/`: Python scripts for generating Manim-based educational animations.
- `tests/`: Automated test suites to verify implementations.

## Development with `uv`

This project uses `uv` as the primary tool for dependency management and task execution.

### Environment Setup
To create a virtual environment and install all dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt -r requirements_dev.txt
```

### Running Tests
To run the project's test suite:
```bash
uv run pytest
```

### Running Animations
To generate animations using Manim:
```bash
uv run python animations/<script_name>.py
```

## Editing Notebooks with Jupyter MCP

The `guides/` directory contains several `.ipynb` files that are the heart of the learning experience. To interact with these notebooks, use the `jupyter` MCP tool.

### Setup a Notebook
Before performing any operations on a notebook, you must initialize the connection:
```python
setup_notebook("guides/es/01. Intro a EduNN.ipynb", server_url="http://localhost:8888")
```

### Common Operations
- **Read Content**: Use `query_notebook(path, "view_source")` to inspect cells.
- **Modify Cells**: Use `modify_notebook_cells` to add, edit, or delete markdown/code cells.
- **Execute Code**: Use `execute_notebook_code` to run existing cells or install packages within the notebook's environment.
- **Add Implementation**: When solving exercises, use `modify_notebook_cells` with `operation='add_code'` or `operation='edit_code'` to provide your implementation.
