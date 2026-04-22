import pytest
import numpy as np
from sklearn.linear_model import LinearRegression

@pytest.fixture(scope="module")
def trained_model():
    """Fixture to train the model once per test module."""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10]) 
    model = LinearRegression()
    model.fit(X, y)
    return model