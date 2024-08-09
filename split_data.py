from scale_data import ready_dataset
from sklearn.model_selection import train_test_split
import numpy as np
# Select Features
features = ready_dataset[['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','A9_Score','A10_Score']]
features_reduced = ready_dataset[['A3_Score','A4_Score','A5_Score','A6_Score','A9_Score',]]
extra=ready_dataset[['age','jundice','A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','A9_Score','A10_Score']]
# Select Target
target = ready_dataset['Class/ASD']

# Set Training and Testing Data
X, X_test, y, y_test = train_test_split(features , target, 
                                                    shuffle = True, 
                                                    test_size=0.2, 
                                                    random_state=1)

def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

baseline_guess = np.median(y)

print('The baseline guess is a score of %0.2f' % baseline_guess)
print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))