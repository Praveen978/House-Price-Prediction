import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os

print("Step 1: Loading your data...")
try:
    df = pd.read_csv('data/house_prices.csv')
    print("Success! Columns found:", df.columns.tolist())
    print(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns\n")
except Exception as e:
    print("Error loading file:", str(e))
    exit()

# Prepare data
X = df.drop(['Property_ID', 'Price'], axis=1)
y = df['Price']

numeric_features    = ['Area', 'Bedrooms', 'Bathrooms', 'Age']
categorical_features = ['Location', 'Property_Type']

preprocessor = ColumnTransformer([
    ('num',  StandardScaler(), numeric_features),
    ('cat',  OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# Models to compare (simple versions to start fast)
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest'    : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost'          : XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

print("Step 2: Training & comparing 3 models...\n")

best_model = None
best_name = ""
best_score = -999

for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    scores = cross_val_score(pipe, X, y, cv=5, scoring='r2', n_jobs=-1)
    mean_r2 = scores.mean()
    
    print(f"{name:18} → Average R² = {mean_r2:.4f}  (std = {scores.std():.4f})")
    
    if mean_r2 > best_score:
        best_score = mean_r2
        best_model = pipe
        best_name = name

print("\n" + "="*60)
print(f"Best model: {best_name}")
print(f"Best cross-validation R²: {best_score:.4f}")
print("="*60 + "\n")

# Train final model on all data & save it
print("Step 3: Training final model and saving...")
best_model.fit(X, y)

os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_model.joblib')
print("Model saved successfully to: models/best_model.joblib")