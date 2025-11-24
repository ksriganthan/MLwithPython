"""
07-5_CV_ApplicationCase3.py

In this script, we will illustrate Application Case 3 of Cross-Validation (CV:
Model Evaluation with HPO using CV.
    1. We split the data set in 2 subsets: Training Set and Validation Set.
    2. We use the Training Sets for
            - cross-validating the hyperparameter k, and then selecting the best k for the k-NN model.
    4. We train the best k-NN model on the Training Set.
    3. We evaluate the best k-NN model on the Validation Set.
"""
# Model Evaluation with HPO mit Cross Validation
################ Preliminaries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold  # For cross-validation
from sklearn.neighbors import KNeighborsClassifier

################ Load the data
diabetes = pd.read_csv('./Teil Machine Learning/Data/diabetes.csv')

################ Prepare the data
X = diabetes.drop(columns=['Outcome']) # Single out the input
y = diabetes['Outcome'] # Single out the target
y = y.values # Convert the target to a numpy array

################ Train/validation split  (80% train, 20% val) -> Kein Test Set ! (ist schon in CV drin)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=23, stratify = y)

################ Hyperparameter Optimization for k-NN with Cross-Validation on the Training Set
k_values = [1, 5, 13, 61, 121, 201, 308]
cv_results = []  # stores the cross-validation recognition rates for each k

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)  # Initialize
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')  # 5-fold CV on the training set
    mean_score = np.mean(scores)  # calculate mean accuracy across folds
    cv_results.append(mean_score) # store the mean score
    print(f"Cross-Validation Recognition Rate for k={k}: {mean_score * 100:.1f}%")

# Find the best k based on cross-validation results
best_k_index = cv_results.index(max(cv_results))
best_k = k_values[best_k_index]
print(f"Best k for k-NN (CV): {best_k} with Cross-Validation Recognition Rate: {cv_results[best_k_index] * 100:.1f}%")

################ Train the best k-NN model on the Training Set
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
print("The best k-NN model has been trained on the Training Set.")

################ Evaluate the best k-NN model on the Validation Set
R_val = best_knn.score(X_val, y_val)
print(f"Validation Set Recognition Rate for the best k-NN model (k={best_k}): {R_val * 100:.1f}%")