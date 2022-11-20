
import requests

import json
# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "hSkuQmck2PyDdxU8ArFuKk6fWcQgVmt3JZNVmDwRhojl"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle as pk

app=Flask(__name__)
model=pk.load(open('CKD.pkl','rb'))

@app.route('/')
def home():
    return render_template('homepage.html')
@app.route('/Prediction',methods=['POST','GET'])
def prediction():
	return render_template('indexpage.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
	return render_template('homepage.html')
@app.route('/predict',methods=['POST'])
def predict():
    input_features=[float(x) for x in request.form.values()]
    
    payload_scoring = {"input_data": [{"field": [['blood_urea','blood_glucose_random','coronary_artery_disease','anemia','pus_cell','red_blood_cells','diabetesmellitus','pedal_edema']], "values": [input_features]}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/6e8755e6-7e54-44e2-99a6-44959f643c2f/predictions?version=2022-11-14', json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    predictions=response_scoring.json()
    pred=predictions['predictions'][0]['values'][0][0]
    if(pred==1):
        return render_template('predictionNo.html')
    else:
        return render_template('predictionYes.html')


if __name__ == '__main__':
    app.run(debug=True)