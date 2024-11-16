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
import csv
import tensorflow as tf
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))
row_list=[]
def fit_and_evaluate(model, my_y, my_x, my_test_x, my_test_y, iteration, starting_point):
    print(model)

    y_train_flat = np.ravel(my_y)
    y_test_flat = np.ravel(my_test_y)    
    model.fit(my_x, y_train_flat)
    model_pred = model.predict(my_test_x)    
    tn, fp, fn, tp = confusion_matrix(my_test_y, model_pred).ravel()
    accuracy = accuracy_score(y_test_flat, model_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    local_results = {}
    local_results['Model']=model
    local_results['Accuracy']=accuracy
    local_results['Sensitivity']=sensitivity
    local_results['Specificity']=specificity
    local_results['Mean Score']=(accuracy+sensitivity+specificity)/3
    local_results['MAE']=mae(y_test_flat, model_pred)
    local_results['Starting point']=starting_point
    local_results['Iteration']=iteration
    row_list.append(local_results)
thresholds = [250, 350, 450, 550]
chunk=50
for threshold in thresholds:
    print("Starting with ")
    print(threshold)
    starting_point=threshold
    i=0
    while threshold < len(X):
        X_chunk = X[:threshold]
        Y_chunk= y[:threshold]
        x_test_chunk=X_test[:threshold]
        y_test_chunk=y_test[:threshold]
        nb_gauss = GaussianNB()
        fit_and_evaluate(nb_gauss, Y_chunk, X_chunk, x_test_chunk, y_test_chunk,i, starting_point)

        decision_tree = DecisionTreeClassifier()
        fit_and_evaluate(decision_tree, Y_chunk, X_chunk, x_test_chunk, y_test_chunk,i, starting_point)

        linear = LinearDiscriminantAnalysis()
        fit_and_evaluate(linear, Y_chunk, X_chunk, x_test_chunk, y_test_chunk,i, starting_point)

        knn = KNeighborsClassifier(n_neighbors=10)
        fit_and_evaluate(knn, Y_chunk, X_chunk, x_test_chunk, y_test_chunk,i, starting_point)

        svm = SVC(C = 1000, gamma = 0.1)
        fit_and_evaluate(svm, Y_chunk, X_chunk, x_test_chunk, y_test_chunk,i, starting_point)

        random_forest = RandomForestClassifier(random_state=60)
        fit_and_evaluate(random_forest, Y_chunk, X_chunk, x_test_chunk, y_test_chunk,i, starting_point)

        nn = MLPClassifier(hidden_layer_sizes=(200,150,100,50),
                                max_iter = 10,activation = 'relu',
                                solver = 'adam')
        fit_and_evaluate(nn, Y_chunk, X_chunk, x_test_chunk, y_test_chunk,i, starting_point)
        threshold+=chunk
        i+=1

results = pd.DataFrame(row_list)

results.to_csv('dynamic_out.csv', index=False)  
    
    


