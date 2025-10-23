from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Initialize app
app = Flask(__name__)

# Load saved model, scaler, and columns
model = joblib.load('model/house_price_model.pkl')
scaler = joblib.load('model/scaler.pkl')
model_columns = joblib.load('model/columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input
        data = {
            'OverallQual': [int(request.form['OverallQual'])],
            'GrLivArea': [float(request.form['GrLivArea'])],
            'GarageCars': [int(request.form['GarageCars'])],
            'FullBath': [int(request.form['FullBath'])],
            'YearBuilt': [int(request.form['YearBuilt'])],
            'TotalBsmtSF': [float(request.form.get('TotalBsmtSF', 0))]
        }

        df = pd.DataFrame(data)

        # One-hot encoding (same as training)
        df = pd.get_dummies(df)

        # Align columns with training columns
        df = df.reindex(columns=model_columns, fill_value=0)

        # Scale features
        df_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(df_scaled)[0]

        return render_template('index.html', prediction_text=f"🏠 Estimated House Price: ${prediction:,.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
