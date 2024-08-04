from flask import Flask, jsonify, request
import flwr as fl
from typing import Optional, List
import numpy as np
import subprocess

app = Flask(__name__)


@app.route('/model')
def get_model():
    state_dict = np.load("weights.npz")
    parameters = fl.common.ndarrays_to_parameters(state_dict.values())
    return jsonify(parameters)


@app.route('/model', methods=['POST'])
def retrain_model():
    answers = request.get_json().answers
    feature = request.get_json().verdict
    s = ','.join(answers)+','+feature
    with open('autism_data_client1.csv', 'w') as f:
        for line in s:
            f.write(line)
    #Do not wait for ML training
    subprocess.Popen('python ml-training.py', shell=True)
    return '', 204