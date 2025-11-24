"""
07-3_CV_ApplicationCase1.py

In this script, we will illustrate Application Case 1 of Cross-Validation (CV:
Model Evaluation w/o HPO using CV.
    - We evaluate the 9-Nearest Neighbors Classifier (k-NN) using CV.
    - To measure prediction performance, we use the Recognition Rate (accuracy).
    - The data set is the diabetes data set.
"""

# Model Evaluation w/o HPO with Cross Validation (Hyperparamater gefixt mit 9 knn)
# ################ Preliminaries

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold  # For cross-validation
from sklearn.neighbors import KNeighborsClassifier

################ Load the data
diabetes = pd.read_csv('./Teil Machine Learning/Data/diabetes.csv')

################ Prepare the data
X = diabetes.drop(columns=['Outcome']) # Single out the input
y = diabetes['Outcome'] # Single out the target
y = y.values # Convert the target to a numpy array

################ Cross-Validation of 9-NN on the entire data set

# For cross-validation, we use the function cross_val_score from sklearn.model_selection.
# This function takes as input the model, the data (X and y), the number of folds (cv=5 for 5-fold CV),
# and the scoring metric (scoring='accuracy' for recognition rate).
# It returns an array of scores, one for each fold.
# We then compute the mean of these scores to get the overall CV recognition rate.

# NOTE:
# - We don't need to split the data into training and validation sets.
# - This is because the function cross_val_score does that internally.

knn = KNeighborsClassifier(n_neighbors=9)  # Initialize
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')  # 5-fold CV on the entire data set
mean_score = np.mean(scores)  # calculate mean accuracy across the folds -> CV Average Test Error Rate
print(f"Cross-Validation Recognition Rate for k=9: {mean_score * 100:.1f}%")

################ Train the model on the entire data set
best_knn = KNeighborsClassifier(n_neighbors=9) # Initialize
best_knn_model = best_knn.fit(X, y) # Fit -> Als nächstes wäre: predict
print("The k-NN model has been trained on the entire data set. It can now be used for predictions.")

# REMEMBER:
# - In k-NN, there is no explicit training phase like in other models.
# - Since k-NN is a lazy learner, training (the "fit"step) simply means storing the data.
# - The model best_knn_model can now be used to make predictions on new data points using the predict method.



