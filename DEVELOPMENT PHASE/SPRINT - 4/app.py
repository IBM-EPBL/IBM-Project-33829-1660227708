import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)
model1 = pickle.load(open("MODEL.pkl",'rb'))

@app.route("/",methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/Predict", methods=['POST'])
def Predict():
    abtest  = int(request.form['abtest'])
    vehicleType = int(request.form['vehicleType'])
    yearOfRegistration  = int(request.form['yearOfRegistration'])
    gearbox = int(request.form['gearbox'])
    powerPS  = int(request.form['powerPS'])
    model  = int(request.form['model'])
    kilometer  = int(request.form['kilometer'])
    monthOfRegistration = int(request.form['monthOfRegistration'])
    fuelType    = int(request.form['fuelType'])
    brand  = int(request.form['brand'])
    notRepairedDamage   = int(request.form['notRepairedDamage'])
    postalCode  = int(request.form['postalCode'])
    pre = [[abtest, vehicleType, yearOfRegistration, gearbox, powerPS, model, kilometer, monthOfRegistration, fuelType, brand, notRepairedDamage ,postalCode]]
    prediction = model1.predict(pre)

    return render_template('predict.html', data=prediction[0])

if __name__=="__main__":
    app.run(debug=True)
