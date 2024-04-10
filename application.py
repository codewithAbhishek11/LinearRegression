from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

model = pickle.load(open('../data/LinearRegression.pkl', 'rb'))
app = Flask(__name__)
cars = pd.read_csv("../data/cleaned_data.csv")

@app.route("/")
def index():
    companies = sorted(cars['company'].unique())
    models = sorted(cars['name'].unique())
    years = sorted(cars['year'].unique())
    fuel_type = sorted(cars['fuel_type'].unique())
    return render_template("index.html", companies=companies, models=models, years = years, fuel_type=fuel_type)

@app.route("/predict", methods=['POST'])
def predict():
    company = request.form.get('company')
    model = request.form.get('models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuelType')
    kilo_driven = int(request.form.get('kiloDriven'))


    prediction = model.predict(pd.DataFrame([[model, company, year,fuel_type,kilo_driven]],
                               columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    print(prediction)

if __name__ == "__main__":
    app.run(debug=True)