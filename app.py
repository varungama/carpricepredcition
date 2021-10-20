from flask import Flask, render_template, request
import requests
import pickle
import os
import pandas as pd
import numpy as np
import sklearn
import joblib

app = Flask(__name__)
model = pickle.load(open('prediction','rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
        
    if request.method == 'POST':
        CarBrand = str(request.form['CarBrand'])
        Model = str(request.form['Model'])
        Year = str(request.form['Year'])
        Miles = int(request.form['Miles'])
        State = str(request.form['State'])
        ExteColor = str(request.form['ExteColor'])
        InterColor = str(request.form['InterColor'])
        style = str(request.form['style'])
        DriveType = str(request.form['DriveType'])
        Transmission = str(request.form['Transmission'])
        Accidents = str(request.form['Accidents'])
        NoOfOwners = str(request.form['NoOfOwners'])
        UseType = str(request.form['UseType'])
        Engine_L = str(request.form['Engine_L'])
        Engine_Gas = str(request.form['Engine_Gas'])
        MPG_cty = int(request.form['MPG_cty'])
        MPG_hwy = int(request.form['MPG_hwy'])

        Miles=round(np.log2(Miles),4)

        fueleconomy= (0.55 * MPG_cty) + (0.45 * MPG_hwy)

        prediction=model.predict(np.array([['CarBrand', 'Model', 'Year', Miles, 'State', 'ExteColor',
       'InterColor', 'style', 'DriveType', Transmission, 'Accidents',
       NoOfOwners, 'UseType', 'Engine_L', 'Engine_Gas', fueleconomy]]))
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction="You can sell the Car at $ {} ".format(output))
    else:
        return render_template('index.html')

if __name__ == "__main__":
   port = int(os.environ.get('PORT' , 5000))
   app.run(host='0.0.0.0', port=port)
