# House Price Prediction

Predicts house prices in Vijayawada using area, bedrooms, bathrooms, age, location, and property type.

## How to Run

1. Install packages
pip install -r requirements.txt
text2. Start the app
python src/web_app.py
text3. Open in browser  
http://127.0.0.1:5000

## What it does
- Web form to enter house details  
- Shows estimated price instantly  
- Uses XGBoost model (best from training)  
- Shows error if invalid input (like 0 area)

## Files
- app.py → main web app  
- templates/index.html → form page  
- models/best_model.joblib → saved model  
- src/model_training.py → training code  
- data/house_prices.csv → dataset
