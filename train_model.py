from split_data import X, X_test, y, y_test
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

# MAE = mean absolute error

scaler = MinMaxScaler(feature_range=(0, 1))
# Fit on the training data
scaler.fit(X)
# Transform both the training and testing data
X = scaler.transform(X)
X_test = scaler.transform(X_test)

def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    # Train the model
    model.fit(X, y)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)
    
    return model_mae

# classification
nb_gauss = GaussianNB()
nb_gauss_mae = fit_and_evaluate(nb_gauss)

decision_tree = DecisionTreeClassifier()
decision_tree_mae = fit_and_evaluate(decision_tree)

linear = LinearDiscriminantAnalysis()
linear_mae = fit_and_evaluate(linear)

knn = KNeighborsClassifier(n_neighbors=10)
knn_mae = fit_and_evaluate(knn)

svm = SVR(C = 1000, gamma = 0.1)
svm_mae = fit_and_evaluate(svm)

random_forest = RandomForestClassifier(random_state=60)
random_forest_mae = fit_and_evaluate(random_forest)

print('NB Gauss mae', nb_gauss_mae)


plt.style.use('fivethirtyeight')
figsize(8, 6)

# Dataframe to hold the results
model_comparison = pd.DataFrame({'model': ['Naive Bayes', 'Support Vector Machine',
                                           'Random Forest', 'Linear Discriminant',
                                            'K-Nearest Neighbors', 'Decision Tree'],
                                 'mae': [nb_gauss_mae, svm_mae, random_forest_mae, 
                                         linear_mae, knn_mae, decision_tree_mae]})

# Horizontal bar chart of test mae
model_comparison.sort_values('mae', ascending = False).plot(x = 'model', y = 'mae', kind = 'barh',
                                                           color = 'red', edgecolor = 'black')

# Plot formatting
plt.ylabel(''); plt.yticks(size = 14); plt.xlabel('Mean Absolute Error'); plt.xticks(size = 14)
plt.title('Model Comparison on Test MAE', size = 20)

plt.show()