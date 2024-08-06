from flask import Flask, jsonify, request
from flask_cors import CORS
import flwr as fl
from typing import Optional, List
import numpy as np
import subprocess

app = Flask(__name__)
CORS(app) 

@app.route('/model')
def get_model():
    state_dict = np.load("weights.npz")
    parameters = fl.common.ndarrays_to_parameters(state_dict.values())
    return jsonify(parameters)


@app.route('/model', methods=['POST'])
def retrain_model():
    answers = request.get_json()['answers']
    feature = 1 if request.get_json()['verdict']=="1" else 0
    s = ','.join(str(x) for x in answers)+','+str(feature) + '\r\n'
    with open('autism_data_client1.csv', 'a') as f:
        f.write(s)
    #Do not wait for ML training
    subprocess.Popen('python ml-training.py', shell=True)
    return '', 204