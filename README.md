# Prediction-of-DON-Concentration

## Project Overview
This repository contains code for predicting **Deoxynivalenol (DON) concentration** in corn samples using **MLPRegressor**. The model is trained on spectral reflectance data to assist in food safety and quality control.



## Dataset
- The dataset consists of spectral readings as features and DON concentration as the target.
- The goal is to predict the DON level based on given input features.

## Dependencies
Ensure you have the following dependencies installed before running the notebook:
pip install pandas numpy scikit-learn matplotlib seaborn shap joblib

## Running the Code

1. Clone this repository:
git clone <repository_url>
2. Navigate to the directory:
cd DON_Prediction
3. Open the Jupyter Notebook:
jupyter notebook ImagoAI_ML_Assignment.ipynb
4. Run all cells to preprocess data, train the model, and visualize results.

## Repository Structure

## DON_Prediction/
│── data/
│   └── MLE-Assignment.csv   # Dataset file
│── models/
│   └── trained_model.pkl    # Saved trained model
│── notebooks/
│   └── ImagoAI_ML_Assignment.ipynb  # Jupyter Notebook
│── README.md                # Documentation
│── requirements.txt         # List of dependencies

## Model Training & Evaluation

Uses MLPRegressor from sklearn.neural_network for regression tasks.
Evaluated using R² Score and Mean Squared Error (MSE).
Results are visualized with scatter plots comparing actual vs. predicted values.

