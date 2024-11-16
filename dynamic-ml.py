from split_data import X, X_test, y, y_test
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
#import tensorflowjs as tfjs
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, PowerTransformer, Normalizer
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))
results = {'Model': [], 'Accuracy': [], 'Sensitivity': [], 'Specificity': [], 'Mean Score':[], 'MAE':[]}
def fit_and_evaluate(model, my_y, my_x, my_test_x, my_test_y, squeeze = False,):
    print(model)

    y_train_flat = np.ravel(my_y)
    y_test_flat = np.ravel(my_test_y)
    if squeeze:
        model.fit(my_x, y_train_flat, epochs=10)
    model.fit(my_x, y_train_flat)
    model_pred = model.predict(my_test_x)
    if squeeze:
        model_pred=tf.squeeze(model_pred)
        model_pred=np.array([1 if x >= 0.5 else 0 for x in model_pred])
    tn, fp, fn, tp = confusion_matrix(my_test_y, model_pred).ravel()
    accuracy = accuracy_score(y_test_flat, model_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    results['Model'].append(model)
    results['Accuracy'].append(accuracy)
    results['Sensitivity'].append(sensitivity)
    results['Specificity'].append(specificity)
    results['Mean Score'].append((accuracy+sensitivity+specificity)/3)
    results['MAE'].append(mae(y_test_flat, model_pred))
thresholds = [250, 350, 450, 550]
chunk=50
for threshold in thresholds:
    print("Starting with ")
    print(threshold)
    while len(X_chunk) < len(X):
        X_chunk = X[:threshold]
        Y_chunk= y[:threshold]
        x_test_chunk=X_test[:threshold]
        y_test_chunk=y_test[:threshold]
        nb_gauss = GaussianNB()
        fit_and_evaluate(nb_gauss, Y_chunk, X_chunk, x_test_chunk, y_test_chunk)

        decision_tree = DecisionTreeClassifier()
        fit_and_evaluate(decision_tree, Y_chunk, X_chunk, x_test_chunk, y_test_chunk)

        linear = LinearDiscriminantAnalysis()
        fit_and_evaluate(linear, Y_chunk, X_chunk, x_test_chunk, y_test_chunk)

        knn = KNeighborsClassifier(n_neighbors=10)
        fit_and_evaluate(knn, Y_chunk, X_chunk, x_test_chunk, y_test_chunk)

        svm = SVC(C = 1000, gamma = 0.1)
        fit_and_evaluate(svm, Y_chunk, X_chunk, x_test_chunk, y_test_chunk)

        random_forest = RandomForestClassifier(random_state=60)
        fit_and_evaluate(random_forest, Y_chunk, X_chunk, x_test_chunk, y_test_chunk)

        nn = MLPClassifier(hidden_layer_sizes=(200,150,100,50),
                                max_iter = 10,activation = 'relu',
                                solver = 'adam')
        fit_and_evaluate(nn, Y_chunk, X_chunk, x_test_chunk, y_test_chunk)
        threshold+=chunk
        print("chunk size:")
        print(threshold)
        print(results)
    
    


