from split_data import X, X_test, y, y_test

import os
import sys
sys.path.insert(0,'/Users/nailyashakirova/opt/anaconda3/lib/python3.8/site-packages/')
import flwr as fl
import ray
import numpy as np
import tensorflow as tf
from functools import partial
import pickle

NUM_CLIENTS = 5
'''
x_split = np.split(x_train, NUM_CLIENTS)
y_split = np.split(y_train, NUM_CLIENTS)
num_data_in_split = x_split[0].shape[0]
train_split = 0.8
x_trains, y_trains, x_tests, y_tests = {}, {}, {}, {}
for idx, (client_x, client_y) in enumerate(zip(x_split, y_split)):
   train_end_idx = int(0.8 * num_data_in_split)
   x_trains[str(idx)] = client_x[:train_end_idx]
   y_trains[str(idx)] = client_y[:train_end_idx]
   x_tests[str(idx)] = client_x[train_end_idx:]
   y_tests[str(idx)] = client_y[train_end_idx:]
'''
class FlowerClient(fl.client.NumPyClient):
   def __init__(self, model, X_train, y_train, X_test, y_test):
       self.model = model
       self.model.build((32, 28, 28, 1))
       self.X_train = X_train
       self.y_train = y_train
       self.X_test = X_test
       self.y_test = y_test
 
   def get_parameters(self, config):
       return self.model.get_weights()
 
   def fit(self, parameters, config):
       self.model.compile("adam", tf.keras.losses.BinaryCrossentropy(from_logits=False,), 
                          metrics=["accuracy"])
       self.model.set_weights(parameters)
       self.model.fit(self.X_train, self.y_train, 
                      epochs=1, batch_size=32, verbose=0)
       return self.model.get_weights(), len(X), {}
 
   def evaluate(self, parameters, config):
       self.model.compile("adam", tf.keras.losses.BinaryCrossentropy(from_logits=False,), 
                          metrics=["accuracy"])
       self.model.set_weights(parameters)
       loss, accuracy = self.model.evaluate(self.X_test, self.y_test, 
                                            batch_size=32, verbose=0)
       return loss, len(self.X_test), {"accuracy": accuracy}

def create_client(
   cid, model, x_trains, y_trains, x_tests, y_tests
) -> FlowerClient:
   return FlowerClient(model, x_trains[cid], y_trains[cid], x_tests[cid], y_tests[cid])

with open("neuralnetworkkeras.pkl", 'rb') as picklefile:
    nnmodel = pickle.load(picklefile)
client_fnc = partial(
   create_client,
   model=nnmodel,
   x_trains=X,
   y_trains=y,
   x_tests=X_test,
   y_tests=y_test,
)

def weighted_average(metrics):
   print(metrics)
   accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
   examples = [num_examples for num_examples, _ in metrics]
   return {"accuracy": int(sum(accuracies)) / int(sum(examples))}
 
# Stwórzmy strategię FedAvg
strategy = fl.server.strategy.FedAvg(
   fraction_fit=1.0,  # Samplujmy 100% dostępnych klientów na trening
   fraction_evaluate=1.0,  # Samplujmy 100% dostępnych klientów na evaluację
   min_fit_clients=5,  # Nie samplujmy mniej niż 5 klientów na trening
   min_evaluate_clients=5,  #Nie samplujmy mniej niż 5 klientów na evaluację
   min_available_clients=5,  # Poczekaj aż 5 klientów jest dostępnych
   evaluate_metrics_aggregation_fn=weighted_average, # Uśrednianie metryk
)

fl.client.start_client(server_address="[::]:8080", client=FlowerClient(nnmodel, X, y, X_test, y_test).to_client())
'''
fl.simulation.start_simulation(
   client_fn=client_fnc,
   num_clients=NUM_CLIENTS,
   config=fl.server.ServerConfig(num_rounds=10),
   strategy=strategy,
   client_resources={"num_cpus": 1, "num_gpus": 0},
   ray_init_args = {"include_dashboard": True}
)
'''