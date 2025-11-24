"""
07-6_CV_ApplicationCase4.py

In this script, we will illustrate Application Case 4 of CV (Cross-Validation):
Model Selection with Hyperparameter Optimization (HPO).
We will optimize and compare the k-Nearest Neighbors Classifier (k-NN) and the Classification Tree model on the diabetes dataset.
    1. We split the data set in 2 subsets: Training Set and Validation Set.
    2. We use the Training Sets for
            - cross-validating k-NN with different values of k, and then selecting the best k.
            - cross-validating a Classification Tree for different values of max_depth, and then selecting the best max_depth.
    3. We compare the best k-NN model and the best Classification Tree model on the Validation Set to find the best of the best.
    4. We train the best model on the entire data set.
    5. Finally, we visualize the best Classification Tree model.

To measure prediction performance, we use the Recognition Rate (accuracy).
"""

# Model Selection with HPO mit Cross Validation
# ################ Preliminaries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score  # For cross-validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

################ Load the data
diabetes = pd.read_csv('./Teil Machine Learning/Data/diabetes.csv')

################ Prepare the data
X = diabetes.drop(columns=['Outcome']) # Single out the input
y = diabetes['Outcome'] # Single out the target
y = y.values # Convert the target to a numpy array

################ Train/validation split  (80% train, 20% val)
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


################ Hyperparameter Optimization for the Classification Tree with Cross-Validation on the Training Set
depth_values = [1, 3, 5, 7, 9, 11, 13]
cv_results = []  # stores the cross-validation recognition rates for each max_depth

for depth in depth_values:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=23)  # Modell initialisieren
    scores = cross_val_score(tree, X_train, y_train, cv=5, scoring='accuracy')  # 5-fache CV
    mean_score = np.mean(scores)  # Durchschnittliche Erkennungsrate berechnen
    cv_results.append(mean_score)
    print(f"Cross-Validation Recognition Rate for max_depth={depth}: {mean_score * 100:.1f}%")

# Find best max_depth
best_depth_index = cv_results.index(max(cv_results))
best_depth = depth_values[best_depth_index]
print(f"Best max_depth for Decision Tree (CV): {best_depth} with Cross-Validation Recognition Rate: {cv_results[best_depth_index] * 100:.1f}%")


################ Train the best k-NN model and the best Classification Tree model on the entire training dataset

best_knn = KNeighborsClassifier(n_neighbors=best_depth)  # Initialize
best_knn_model = best_knn.fit(X_train, y_train)

best_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=23)  # Initialize
best_tree_model = best_tree.fit(X_train, y_train)


################ Select the best of the best

# We compare the Recognition Rates of the best k-NN with the best Classification Tree on the Validation Set
R_best_knn = best_knn.score(X_val, y_val)
R_best_tree = best_tree.score(X_val, y_val)
print(f"Best k-NN Validation Recognition Rate: {R_best_knn * 100:.1f}%")
print(f"Best Classification Tree Validation Recognition Rate: {R_best_tree * 100:.1f}%")

## Determine the best of the best
if R_best_knn > R_best_tree:
    print(f"The best model is k-NN with k={best_k} achieving a Validation Recognition Rate of {R_best_knn * 100:.1f}%")
else:
    print(f"The best model is Classification Tree with max_depth={best_depth} achieving a Validation Recognition Rate of {R_best_tree * 100:.1f}%")


