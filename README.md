# 🏠 House Price Prediction Web App  

This project is a Machine Learning-based web app that predicts house prices using real-world data. Built with Python, Flask, Pandas, NumPy, and Scikit-learn, it covers the full pipeline from data preprocessing to real-time prediction. The model, trained on the Kaggle House Prices dataset using Gradient Boosting Regressor, delivers accurate results after fine-tuning with GridSearchCV. The simple HTML-CSS frontend takes property details as input and displays instant predictions via a Flask backend that connects to stored model and scaler files.  

## 🚀 Features  
- Predict house prices instantly through a simple web form  
- Clean and responsive UI built with HTML and CSS  
- Feature scaling for improved prediction accuracy  
- End-to-end implementation of model training and deployment  

## 🧠 Tech Stack  
Python · Flask · Pandas · NumPy · Scikit-learn · Joblib · HTML5 · CSS3  

## ⚙️ How to Run  
1. Clone this repository  
2. Navigate to the project folder  
3. (Optional) Create a virtual environment: `python -m venv venv` and activate it  
4. Install dependencies: `pip install -r requirement.txt`  
5. Run the Flask app: `python app.py`  
6. Open your browser and visit: `http://127.0.0.1:5000/`  

## 🗂️ Project Structure  
app.py – Flask backend logic  
model/ – Contains trained model, scaler, and columns files  
templates/ – HTML templates for UI  
static/ – CSS stylesheets  
Notebook/ – Jupyter Notebook used for model training  
README.md – Project documentation  

## 📌 Note  
This project is created for learning and demonstration purposes. You can retrain the model using your own dataset or modify the interface for additional features.
