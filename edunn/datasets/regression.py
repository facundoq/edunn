import numpy as np
from . import basepath

regression_basepath = basepath / "regression_data"


def load_regression_dataset(filename, output_values=1):
    data = np.loadtxt(regression_basepath / filename, skiprows=1, delimiter=",")
    x = data[:, :-output_values]
    y = data[:, -output_values:]
    return x, y


# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=train.csv
# TODO cleanup
houses = lambda: load_regression_dataset("houses.csv")

study1d = lambda: load_regression_dataset("study1d.csv")
study2d = lambda: load_regression_dataset("study2d.csv", output_values=2)

# https://www.kaggle.com/c/boston-housing
boston = lambda: load_regression_dataset("boston_housing.csv")

# https://archive.ics.uci.edu/ml/datasets/wine+quality
wine_red = lambda: load_regression_dataset("wine_red.csv")
wine_white = lambda: load_regression_dataset("wine_white.csv")

# https://www.kaggle.com/quantbruce/real-estate-price-prediction
real_state = lambda: load_regression_dataset("real_state.csv")

# https://www.kaggle.com/mirichoi0218/insurance
insurance = lambda: load_regression_dataset("insurance.csv")

loaders = {
    "study1d": study1d,
    "study2d": study2d,
    "boston": boston,
    "wine_red": wine_red,
    "wine_white": wine_white,
    "insurance": insurance,
    "real_state": real_state,
}
