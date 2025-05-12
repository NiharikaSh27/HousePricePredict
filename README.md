**House Price Prediction**
* This is a Flask-based web application that predicts house prices using a machine learning model trained on the Ames Housing dataset (train.csv). It uses an XGBoost regressor to estimate prices based on key house features and provides a user-friendly interface for inputting data, viewing predictions, and exploring model performance through visualizations.
About This File
* This README.md file provides instructions for setting up, running, and using the House Price Prediction project. Download and save it as README.md in your project directory (HousePricePrediction/) to serve as documentation for the project.
Project Overview
* The application enables users to input house characteristics (e.g., overall quality, living area) to predict sale prices.
*  It also includes visualizations to assess model performance: Actual vs. Predicted Prices Scatter Plot: 
    * Shows how closely predictions match actual prices.
    * Feature Importance Bar Plot: Highlights which features drive predictions.
    * Prediction Error Distribution Histogram: Displays the distribution of prediction errors.

* Model metrics (Mean Squared Error and R-squared) are displayed in styled cards, making the app accessible to real estate enthusiasts, data scientists, and developers.
  
**Project Structure**

HousePricePrediction/
  * app.py              # Flask application with model training and visualization logic
  * train.csv           # Ames Housing dataset
  * model.pkl           # Trained XGBoost model
  * scaler.pkl          # StandardScaler for feature scaling
  * static/             # Folder for visualization images
      * actual_vs_predicted.png
      * feature_importance.png
      * error_distribution.png
  * templates/
      * index.html      # HTML template for the web interface

* Features
  * Input Features:
    * Predicts house prices using six features:
    * OverallQual: Overall material and finish quality (1–10).
    * rLivArea: Above-ground living area in square feet.
    * GarageCars: Number of cars the garage can hold.
    * TotalBsmtSF: Total basement area in square feet.
    * FullBath: Number of full bathrooms.
    * YearBuilt: Year the house was built.

* Model: Trained XGBoost regressor with MSE and R-squared metrics displayed in styled cards.
* Web Interface: Allows users to input feature values, view predictions, and see visualizations. Errors (e.g., invalid inputs) are displayed clearly.
* Visualizations:
  * Scatter plot of actual vs. predicted prices.
  * Bar plot of feature importance.
  * Histogram of prediction errors with a KDE curve.


* Frontend: Responsive design with CSS styling, including metric cards and image displays.

* Prerequisites
  * Python 3.12 or compatible
  * Required Python packages: pandas, numpy, scikit-learn, xgboost, flask, joblib, matplotlib, seaborn
  * The train.csv dataset (provided)


* Access the Web Interface:Open a browser and go to http://127.0.0.1:5001. You will see:

  * A form to input feature values.
  * Metric cards showing MSE and R-squared.
  * Visualizations of model performance (scatter plot, feature importance, error distribution).

* Usage

  * Input Values:Enter values for the six features:

    * OverallQual (1–10, e.g., 7)
    * GrLivArea (square feet, e.g., 1500)
    * GarageCars (number, e.g., 2)
    * TotalBsmtSF (square feet, e.g., 1000)
    * FullBath (number, e.g., 2)
    * YearBuilt (year, e.g., 2000)


    * Predict:Click "Predict Price" to see the estimated house price.

* View Metrics and Visualizations:

  * MSE and R-squared: Displayed in cards, indicating model accuracy and fit.
    
* Visualizations:
    * Actual vs. Predicted: Points near the diagonal line show accurate predictions.
    * Feature Importance: Shows which features (e.g., GrLivArea) are most influential.
    * Error Distribution: A bell-shaped curve near 0 indicates good performance.
    
   
* Error Handling:Invalid inputs (e.g., non-numeric values) display an error message.


**Example**
* Enter the following values for a prediction:

  * OverallQual: 7
  * GrLivArea: 1500
  * GarageCars: 2
  * TotalBsmtSF: 1000
  * FullBath: 2
  * YearBuilt: 2000

* Click "Predict Price" to see the predicted house price (e.g., ~$200,000, depending on the model). Scroll down to view visualizations and metrics.

Notes

* The model is trained on train.csv and saved as model.pkl. 
* Visualizations are generated at startup and saved in static/, ensuring fast page loads.
* The application is intended for local use in debug mode (debug=True). For production, configure Flask with a production server (e.g., Gunicorn).
