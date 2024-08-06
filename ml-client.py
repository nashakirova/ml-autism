
import sys

import flwr as fl
import numpy as np
import tensorflow as tf
from functools import partial
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

print('ML CLIENT starting')
class FlowerClient(fl.client.NumPyClient):
   def __init__(self, model):
        dataset = pd.read_csv('./autism_data_client1.csv')
        dataset.head()
        features=dataset[['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']]
        target=dataset['autism']
        print('FEATURES AND TARGET initialised')
        X, X_test, y, y_test = train_test_split(features , target, 
                                                            shuffle = True, 
                                                            test_size=0.2, 
                                                            random_state=1)
        self.model = model
        self.model.build((32, 28, 28, 1))
        self.X_train = X
        self.y_train = y
        self.X_test = X_test
        self.y_test = y_test
 
   def get_parameters(self, config):
        print(self.model.get_weights())
        return self.model.get_weights()
 
   def fit(self, parameters, config):
       self.model.compile("adam", tf.keras.losses.BinaryCrossentropy(from_logits=False,), 
                          metrics=["accuracy"])
       print('TYPE OF', type(parameters))
       self.model.set_weights(parameters)
       self.model.fit(self.X_train, self.y_train, 
                      epochs=1, batch_size=32, verbose=0)
       return self.model.get_weights(), len(self.X_train), {}
 
   def evaluate(self, parameters, config):
       self.model.compile("adam", tf.keras.losses.BinaryCrossentropy(from_logits=False,), 
                          metrics=["accuracy"])
       self.model.set_weights(parameters)
       loss, accuracy = self.model.evaluate(self.X_test, self.y_test, 
                                            batch_size=32, verbose=0)
       return loss, len(self.X_test), {"accuracy": accuracy}

def create_client(cid) -> FlowerClient:
    tf_model = tf.keras.Sequential()
    tf_model.add(tf.keras.layers.Dense(200, input_shape=(10,), activation='relu'))
    tf_model.add(tf.keras.layers.Dense(150, input_shape=(10,), activation='relu'))
    tf_model.add(tf.keras.layers.Dense(100, input_shape=(10,), activation='relu'))
    tf_model.add(tf.keras.layers.Dense(50, input_shape=(10,), activation='relu'))
    tf_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return FlowerClient(tf_model).to_client()

fl.client.start_client(server_address="[::]:8080", client_fn=create_client)