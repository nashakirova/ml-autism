
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Optional, List
#This is to validate the model against autism_screening

tf_model = tf.keras.Sequential()
tf_model.add(tf.keras.layers.Dense(200, input_shape=(10,), activation='relu'))
tf_model.add(tf.keras.layers.Dense(150, input_shape=(10,), activation='relu'))
tf_model.add(tf.keras.layers.Dense(100, input_shape=(10,), activation='relu'))
tf_model.add(tf.keras.layers.Dense(50, input_shape=(10,), activation='relu'))
tf_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

file = np.load("weights.npz")

files = file.files
npArray = []
for item in files:
    npArray.append(file[item])
tf_model.set_weights(npArray)
tf_model.compile("adam", tf.keras.losses.BinaryCrossentropy(from_logits=False,), 
                          metrics=["accuracy"])
from split_data import X_test, y_test
loss, accuracy = tf_model.evaluate(X_test, y_test, 
                                            batch_size=32, verbose=0)
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

print(accuracy)
