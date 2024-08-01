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
def fit_and_evaluate(model, squeeze = False):
    print(model)
    # Train the model
    model.fit(X, y)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    # Calculate true positive, false positive, true negative, false negative
    if squeeze:
        model_pred=tf.squeeze(model_pred)
        model_pred=np.array([1 if x >= 0.5 else 0 for x in model_pred])
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

nn = MLPClassifier(hidden_layer_sizes=(200,150,100,50),
                        max_iter = 500,activation = 'relu',
                        solver = 'adam')
fit_and_evaluate(nn)

# Now, to use it with the tensorflow, I need to implement my own MLP classifier using keras module.

# tf_model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(8,)),
#     tf.keras.layers.Dense(150, activation='softmax'),
#     tf.keras.layers.Dense(100, activation='softmax'),
#     tf.keras.layers.Dense(50, activation='softmax')
# ])
import time
start_time = time.time()
tf_model = tf.keras.Sequential()
tf_model.add(tf.keras.layers.Dense(200, input_shape=(10,), activation='relu'))
tf_model.add(tf.keras.layers.Dense(150, input_shape=(10,), activation='relu'))
tf_model.add(tf.keras.layers.Dense(100, input_shape=(10,), activation='relu'))
tf_model.add(tf.keras.layers.Dense(50, input_shape=(10,), activation='relu'))
tf_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

tf_model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False,),
              metrics=['accuracy'])
fit_and_evaluate(tf_model, True)
print("NAILIAAAA--- %s seconds ---" % (time.time() - start_time))


scores = pd.DataFrame(results)

sorted_df = scores.sort_values(by='MAE', ascending=True)
print('CLASSIFIERS')
print(sorted_df)

scaling_techniques = [
    StandardScaler(), 
    RobustScaler(), 
    QuantileTransformer(), 
    MinMaxScaler(),
    MaxAbsScaler(),
    PowerTransformer(),
]

fig, axs = plt.subplots(2, 3, figsize=(15, 6))

# Loop through each subplot and scaling technique
for i, ax in enumerate(axs.flat):
    scaler = scaling_techniques[i]
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.concat([pd.DataFrame(X_scaled)], axis=1)
    mask = np.triu(np.ones_like(df_scaled.corr(), dtype=bool))
    sns.heatmap(df_scaled.corr(), cmap='coolwarm', annot=False, mask=mask, ax=ax)
    ax.set_title(f'{type(scaler).__name__}')
plt.tight_layout()
#plt.show()

X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=25)
scaler = StandardScaler()

names = ['DecisionTreeClassifier', 'RandomForestClassifier', 'LinearDiscriminantAnalysis',
         'SupportVectorMachine', 'KNearestNeighbor','NaiveBayes', 'MLPClassifier', 'TF MLP Classifier']
models = [
    DecisionTreeClassifier(max_depth=50, min_samples_leaf=60),
    RandomForestClassifier(n_estimators=250,max_depth=10, min_samples_leaf=25),
    LinearDiscriminantAnalysis(), SVC(C=1.0, kernel='rbf', degree=3, gamma='scale'),
    KNeighborsClassifier(n_neighbors=3),GaussianNB(), MLPClassifier(hidden_layer_sizes=(200,150,100,50),
                        max_iter = 500,activation = 'relu',
                        solver = 'adam'), tf_model
]

results_df = pd.DataFrame(columns=[type(scaler).__name__], index=names)
trained_models = []
for counter, model in enumerate(models):
    # Convert y_train and y_test to 1-dimensional arrays
    y_train_flat = np.ravel(y_train)
    y_test_flat = np.ravel(y_test)
    # Train
    model.fit(X_train, y_train_flat)
    trained_models.append(model)
    y_pred=model.predict(X_test)
    # binarize the prediction
    if counter == 7:
        y_pred=tf.squeeze(y_pred)
        y_pred=np.array([1 if x >= 0.5 else 0 for x in y_pred])
    print(names[counter], metrics.accuracy_score(y_test_flat, y_pred))
    results_df.loc[names[counter], type(scaler).__name__] = metrics.accuracy_score(y_test_flat, y_pred)    

print(results_df)
chosen_model = trained_models[7] #NN
#tfjs.converters.save_keras_model(chosen_model, 'nn_keras')

with open('neuralnetworkkeras.pkl','wb') as f:
    pickle.dump(chosen_model,f,protocol=2)