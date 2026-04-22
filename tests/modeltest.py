import numpy as np
from sklearn.linear_model import LinearRegression

def test_model_logic():
    X = np.array([[1],[2],[3]])
    y = np.array([2,4,6])
    model = LinearRegression().fit(X, y)
    assert np.isclose(model.predict([[4]])[0], 8.0)