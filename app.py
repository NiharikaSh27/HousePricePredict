import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from flask import Flask, request, render_template
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Flask app
app = Flask(__name__)

# Ensure static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Load and preprocess dataset
def load_and_preprocess_data():
    df = pd.read_csv('train.csv')
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    target = 'SalePrice'
    
    df = df[features + [target]].copy()
    df['GarageCars'] = df['GarageCars'].fillna(0)
    df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
    
    X = df[features]
    y = df[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, features

# Train XGBoost model and generate visualizations
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Generate visualizations
    generate_visualizations(y_test, y_pred, model, X_test, feature_names)
    
    return model, mse, r2, X_test, y_test

# Generate visualizations
def generate_visualizations(y_test, y_pred, model, X_test, feature_names):
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    # 1. Actual vs Predicted Prices Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Actual vs Predicted House Prices')
    plt.tight_layout()
    plt.savefig('static/actual_vs_predicted.png')
    plt.close()
    
    # 2. Feature Importance Bar Plot
    plt.figure(figsize=(8, 6))
    feature_importance = model.feature_importances_
    sns.barplot(x=feature_importance, y=feature_names, palette='viridis')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance in House Price Prediction')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    plt.close()
    
    # 3. Prediction Error Distribution
    errors = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=30, kde=True, color='purple')
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.tight_layout()
    plt.savefig('static/error_distribution.png')
    plt.close()

# Load data and train model
X, y, scaler, feature_names = load_and_preprocess_data()
model, mse, r2, X_test, y_test = train_model(X, y)

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Flask routes
@app.route('/')
def home():
    return render_template('index.html', mse=mse, r2=r2, features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[feat]) for feat in feature_names]
        input_array = np.array([input_data])
        
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        
        return render_template('index.html', prediction=prediction, mse=mse, r2=r2, features=feature_names)
    except Exception as e:
        return render_template('index.html', error=str(e), mse=mse, r2=r2, features=feature_names)

if __name__ == '__main__':
    app.run(debug=True, port=5001)