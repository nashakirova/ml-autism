from flask import Flask, jsonify, request
from flask_cors import CORS
import flwr as fl
from typing import Optional, List
import numpy as np
import subprocess
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user

app = Flask(__name__)
CORS(app) 
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
app.config["SECRET_KEY"] = "SuperSecretKey"
db = SQLAlchemy()
login_manager = LoginManager()
login_manager.init_app(app)
class Users(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(250), unique=True,
                         nullable=False)
    password = db.Column(db.String(250),
                         nullable=False)
db.init_app(app)
with app.app_context():
    db.create_all()

@login_manager.user_loader
def loader_user(user_id):
    return Users.query.get(user_id)

def validateUser(username, password):
        user = Users.query.filter_by(
            username=username).first()
        if user.password == password:
            login_user(user)
            return True
        return False

@app.route('/model')
def get_model():
        file = np.load("weights.npz")

        files = file.files
        npArray = []
        for item in files:
                npArray.append(file[item])
        return jsonify({"parameters": npArray})


@app.route('/model', methods=['POST'])
def retrain_model():
        validateUser(request.get_json()['username'], request.get_json()['password'])
        answers = request.get_json()['answers']
        feature = 1 if request.get_json()['verdict']=="1" else 0
        s = ','.join(str(x) for x in answers)+','+str(feature) + '\r\n'
        with open('autism_data_client1.csv', 'a') as f:
                f.write(s)
        #Do not wait for ML training
        subprocess.Popen('python ml-training.py', shell=True)
        return '', 204