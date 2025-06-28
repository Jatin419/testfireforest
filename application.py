import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle

from flask import Flask, render_template, request

application = Flask(__name__)
app = application

# Load the model and scaler
ridge_model = pickle.load(open(r'models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open(r'models/scaler.pkl', 'rb'))

# Home Route
@app.route("/")
def index():
    return render_template("index.html")

# Prediction Route
@app.route("/predict_data", methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
        # Extract data from the form
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])

        # Prepare the input for the model
        input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        scaled_data = standard_scaler.transform(input_data)

        # Make prediction
        prediction = ridge_model.predict(scaled_data)

        return render_template('home.html', result=round(prediction[0], 2))

    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
