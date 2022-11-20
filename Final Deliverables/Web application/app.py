import pickle as pk

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app=Flask(__name__)
model=pk.load(open('CKD.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction',methods=['POST','GET'])
def prediction():
	return render_template('index.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
	return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    input_features=[float(x) for x in request.form.values()]
    features_value=[np.array(input_features)]
    features_name=['blood_urea','blood glucose random','coronary_artery_disease','anemia','pus_cell','red_blood_cells','diabetesmellitus','pedal_edema']
    df=pd.DataFrame(features_value,columns=features_name)
    output=model.predict(df)
    if(output==0):
        return render_template('predictionNo.html')
    else:
        return render_template('predictionYes.html')


if __name__ == '__main__':
    app.run(debug=True)