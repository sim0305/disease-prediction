from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import pandas as pd
import pickle
import numpy as np

# Load the models
diabetes_model = pickle.load(open('diabetes1.pkl', 'rb'))
heart_model = pickle.load(open('heart_disease_model.sav', 'rb'))
liver_model = pickle.load(open('liver.pkl', 'rb'))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
bootstrap = Bootstrap(app)

# Default route for the dashboard
@app.route('/')
def dashboard():
    return render_template("dashboard.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/liver")
def liver():
    return render_template("liver.html")

@app.route('/predictdiabetes', methods=['POST'])
def predictdiabetes():
    if request.method == 'POST':
        data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['bloodpressure']),
            float(request.form['skinthickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]
        data = np.array([data])
        prediction = diabetes_model.predict(data)
        if prediction == 1:
            result = "Patient has a high risk of Diabetes, please consult your doctor."
        else:
            result = "Patient has a low risk of Diabetes."
        return render_template('diab_result.html', prediction_text=result)

@app.route('/predictheart', methods=['POST'])
def predictheart():
    if request.method == 'POST':
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]

        prediction = heart_model.predict(features_value)
        if prediction == 1:
            result = "Patient has a high risk of Heart Disease, please consult your doctor."
        else:
            result = "Patient has a low risk of Heart Disease."
        return render_template('heart_result.html', prediction_text=result)

@app.route('/predictliver', methods=['POST'])
def predictliver():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        data = np.array([to_predict_list])
        prediction = liver_model.predict(data)

        if prediction == 1:
            result = "Patient has a high risk of Liver Disease, please consult your doctor."
        else:
            result = "Patient has a low risk of Liver Disease."
        return render_template("liver_result.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
