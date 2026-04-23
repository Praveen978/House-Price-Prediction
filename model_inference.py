# Prediction functions
# Load model and make predictions here
# src/model_inference.py
"""
Module for loading the trained model and making price predictions.
"""

import joblib
import pandas as pd

# Global variables (loaded once)
MODEL_PATH = 'models/best_model.joblib'
model = None

def load_model():
    """
    Load the saved model from disk.
    Call this once when the app starts.
    """
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from: {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        model = None


def predict_price(input_data):
    """
    Make a house price prediction from input features.
    
    Parameters:
        input_data (dict): Dictionary with keys:
            'Area', 'Bedrooms', 'Bathrooms', 'Age', 'Location', 'Property_Type'
    
    Returns:
        float: Predicted price (rounded to nearest integer)
    
    Raises:
        RuntimeError: If model is not loaded
    """
    if model is None:
        load_model()
        if model is None:
            raise RuntimeError("Model could not be loaded")

    # Convert input dict to DataFrame (must match training column names)
    df_input = pd.DataFrame([input_data])

    # Make prediction
    predicted_price = model.predict(df_input)[0]

    return round(predicted_price)


# Optional: Load model when this module is imported (for Flask app)
if __name__ != '__main__':
    load_model()