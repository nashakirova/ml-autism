from split_data import X, X_test, y, y_test
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
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

# MAE = mean absolute error

scaler = MinMaxScaler(feature_range=(0, 1))
# Fit on the training data
scaler.fit(X)
# Transform both the training and testing data
X = scaler.transform(X)
X_test = scaler.transform(X_test)

def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

results = {'Model': [], 'Accuracy': [], 'Sensitivity': [], 'Specificity': [], 'Mean Score':[], 'MAE':[]}
# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    # Train the model
    model.fit(X, y)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    # Calculate true positive, false positive, true negative, false negative
    tn, fp, fn, tp = confusion_matrix(y_test, model_pred).ravel()

    # Calculate accuracy
    accuracy = accuracy_score(y_test, model_pred)
    
    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
        # Store the results in the dictionary
    results['Model'].append(model)
    results['Accuracy'].append(accuracy)
    results['Sensitivity'].append(sensitivity)
    results['Specificity'].append(specificity)
    results['Mean Score'].append((accuracy+sensitivity+specificity)/3)
    results['MAE'].append(mae(y_test, model_pred))

# classification
nb_gauss = GaussianNB()
fit_and_evaluate(nb_gauss)

decision_tree = DecisionTreeClassifier()
fit_and_evaluate(decision_tree)

linear = LinearDiscriminantAnalysis()
fit_and_evaluate(linear)

knn = KNeighborsClassifier(n_neighbors=10)
fit_and_evaluate(knn)

svm = SVC(C = 1000, gamma = 0.1)
fit_and_evaluate(svm)

random_forest = RandomForestClassifier(random_state=60)
fit_and_evaluate(random_forest)


scores = pd.DataFrame(results)

sorted_df = scores.sort_values(by='MAE', ascending=True)
print(sorted_df)