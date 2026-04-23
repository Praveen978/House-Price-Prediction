# src/web_app.py - House Price Prediction Web App

from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder='../templates')  # ← This fixes TemplateNotFound

# Debug prints to confirm paths (you can remove later)
print("Current working directory:", os.getcwd())
print("Looking for templates in:", os.path.abspath(app.template_folder))
print("Does templates folder exist?", os.path.isdir(app.template_folder))
print("Does index.html exist?", os.path.exists(os.path.join(app.template_folder, 'index.html')))

# Load the model
model = None
try:
    model = joblib.load('models/best_model.joblib')
    print("MODEL LOADED SUCCESSFULLY")
except Exception as e:
    print("MODEL LOAD FAILED:", str(e))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            area = float(request.form['area'])
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])
            age = int(request.form['age'])
            location = request.form['location']
            property_type = request.form['property_type']

            # Input validation
            if area <= 0:
                raise ValueError("Area must be greater than 0 sqft")
            if bedrooms < 1:
                raise ValueError("At least 1 bedroom is required")
            if bathrooms < 1:
                raise ValueError("At least 1 bathroom is required")
            if age < 0:
                raise ValueError("Age cannot be negative")

            # Create input DataFrame (exact column names your model expects)
            input_data = pd.DataFrame([{
                'Area': area,
                'Bedrooms': bedrooms,
                'Bathrooms': bathrooms,
                'Age': age,
                'Location': location,
                'Property_Type': property_type
            }])

            # Make prediction
            price = model.predict(input_data)[0]
            prediction = f"₹{int(round(price)):,}"

        except ValueError as ve:
            error = str(ve)
        except Exception as e:
            error = f"Error: {str(e)}. Please check all fields."

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)