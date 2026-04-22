import numpy as np

def test_prediction(trained_model):
    # The 'trained_model' argument automatically comes from conftest.py
    prediction = trained_model.predict([[6]])
    assert np.isclose(prediction[0], 12.0)

def test_slope(trained_model):
    # Verify the slope is 2 (y = 2x)
    assert np.isclose(trained_model.coef_[0], 2.0)