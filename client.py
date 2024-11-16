from split_data import X, X_test, y, y_test
import sys

import flwr as fl
import numpy as np
import tensorflow as tf
from functools import partial
import pickle

print('CLIENT IS HERE')
NUM_CLIENTS = 8
#563
X_divided = X[:len(X)-3]
y_divided=y[:len(y)-3]
x_split = np.split(X_divided, NUM_CLIENTS)
y_split = np.split(y_divided, NUM_CLIENTS)
num_data_in_split = x_split[0].shape[0]
train_split = 0.8
x_trains, y_trains, x_tests, y_tests = {}, {}, {}, {}
for idx, (client_x, client_y) in enumerate(zip(x_split, y_split)):
   train_end_idx = int(train_split * num_data_in_split)
   x_trains[str(idx)] = client_x[:train_end_idx]
   y_trains[str(idx)] = client_y[:train_end_idx]
   x_tests[str(idx)] = client_x[train_end_idx:]
   y_tests[str(idx)] = client_y[train_end_idx:]
class FlowerClient(fl.client.NumPyClient):
   def __init__(self, cid, model):
       self.model = model
       self.model.build((32, 28, 28, 1))
       self.cid=cid
       self.X_train = x_trains[cid]
       self.y_train = y_trains[cid]
       self.X_test = x_tests[cid]
       self.y_test = y_tests[cid]
 
   def get_parameters(self, config):
      return self.model.get_weights()
 
   def fit(self, parameters, config):
       self.model.compile("adam", tf.keras.losses.BinaryCrossentropy(from_logits=False,), 
                          metrics=["accuracy"])
       self.model.set_weights(parameters)
       self.model.fit(self.X_train, self.y_train, 
                      epochs=10, batch_size=32, verbose=0)
       return self.model.get_weights(), len(X), {}
 
   def evaluate(self, parameters, config):
       self.model.compile("adam", tf.keras.losses.BinaryCrossentropy(from_logits=False,), 
                          metrics=["accuracy"])
       self.model.set_weights(parameters)
       loss, accuracy = self.model.evaluate(self.X_test, self.y_test, 
                                            batch_size=32, verbose=0)
       return loss, len(self.X_test), {"accuracy": accuracy}

def create_client(cid) -> FlowerClient:
   return FlowerClient(cid, tf_model).to_client()

with open("neuralnetworkkeras.pkl", 'rb') as picklefile:
      #nnmodel = pickle.load(picklefile)
      tf_model = tf.keras.Sequential()
      tf_model.add(tf.keras.layers.Dense(200, input_shape=(10,), activation='relu'))
      tf_model.add(tf.keras.layers.Dropout(0.2))
      tf_model.add(tf.keras.layers.Dense(150, input_shape=(10,), activation='relu'))
      tf_model.add(tf.keras.layers.Dropout(0.2))
      tf_model.add(tf.keras.layers.Dense(100, input_shape=(10,), activation='relu'))
      tf_model.add(tf.keras.layers.Dropout(0.2))
      tf_model.add(tf.keras.layers.Dense(50, input_shape=(10,), activation='relu'))
      tf_model.add(tf.keras.layers.Dropout(0.2))
      tf_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# client_fnc = partial(
#    create_client,
#    model=tf_model,
#    x_trains=x_trains,
#    y_trains=y_trains,
#    x_tests=x_tests,
#    y_tests=y_tests,
# )
#fl.client.start_client(server_address="[::]:8080", client_fn=create_client)
#app = fl.client.ClientApp(client_fn=create_client)
def weighted_average(metrics):
   accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
   examples = [num_examples for num_examples, _ in metrics]
   return {"accuracy": int(sum(accuracies)) / int(sum(examples))}
strategy = fl.server.strategy.FedAvg(
   fraction_fit=1.0,  
   fraction_evaluate=1.0, 
   min_fit_clients=2,  
   min_evaluate_clients=2, 
   min_available_clients=2, 
   evaluate_metrics_aggregation_fn=weighted_average
)

dp_strategy = fl.server.strategy.DifferentialPrivacyServerSideFixedClipping(
    strategy, 1.0, 1.0, 2
)

fl.simulation.start_simulation(
   client_fn=create_client,
   num_clients=NUM_CLIENTS,
   config=fl.server.ServerConfig(num_rounds=8),
   strategy=strategy,
   client_resources={"num_cpus": 1, "num_gpus": 0},
)