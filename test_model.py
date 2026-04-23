# tests/test_model.py
# Basic unit tests for the house price prediction model

import pytest
import joblib
import pandas as pd
import numpy as np
from web_app import model  # adjust import if your model is loaded in web_app.py

# Skip all tests if model is not loaded (for safety)
pytestmark = pytest.mark.skipif(model is None, reason="Model not loaded")


def test_model_is_loaded():
    """Test that the model is successfully loaded."""
    assert model is not None, "Model should be loaded from joblib file"


def test_model_has_predict_method():
    """Test that the loaded model has a predict method."""
    assert hasattr(model, 'predict'), "Loaded model should have a 'predict' method"


def test_prediction_on_valid_input():
    """Test prediction shape and type on a valid dummy input."""
    dummy_input = pd.DataFrame([{
        'Area': 1500.0,
        'Bedrooms': 3,
        'Bathrooms': 2,
        'Age': 5,
        'Location': 'City Center',
        'Property_Type': 'Apartment'
    }])

    prediction = model.predict(dummy_input)

    assert isinstance(prediction, np.ndarray), "Prediction should be a numpy array"
    assert prediction.shape == (1,), "Prediction should return a single value"
    assert prediction[0] > 0, "Predicted price should be positive"


def test_prediction_on_extreme_input():
    """Test model doesn't crash on large values (basic sanity check)."""
    extreme_input = pd.DataFrame([{
        'Area': 10000.0,
        'Bedrooms': 10,
        'Bathrooms': 8,
        'Age': 0,
        'Location': 'City Center',
        'Property_Type': 'Villa'
    }])

    prediction = model.predict(extreme_input)

    assert prediction[0] > 0, "Model should handle extreme inputs without crashing"


def test_invalid_input_raises_error():
    """Test that invalid input raises appropriate error."""
    invalid_input = pd.DataFrame([{
        'Area': 'invalid_string',  # wrong type
        'Bedrooms': 3,
        'Bathrooms': 2,
        'Age': 5,
        'Location': 'City Center',
        'Property_Type': 'Apartment'
    }])

    with pytest.raises(Exception):  # or ValueError, depending on your pipeline
        model.predict(invalid_input)