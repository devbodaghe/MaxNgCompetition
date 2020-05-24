import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
import pandas as pd
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier


app = Flask(__name__)
model = pickle.load(open('assets/model.pkl', 'rb'))
columns = ['travel_time', 'month', 'weekday', 'rainfall', 'periods', 'busy_days', 'holidays']

with open("assets/data.json", "r") as read_file:
    le_json = json.load(read_file)
le = json.loads(le_json)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    output = None
    int_features = {i: int(j) for i, j in zip(columns, request.form.values())}
    final_features = pd.DataFrame([int_features])
    prediction = model.predict(final_features)
    for i in le['travel_from']:
        if le['travel_from'][i] == prediction:
            output = i
            break

    return render_template('index.html', prediction_text="{}".format(output), url="https://cf1d35f1.ngrok.io")


if __name__ == "__main__":
    app.run(debug=True)