"""
07-2_VSA_ApplicationCase4.py

In this script, we will illustrate Application Case 4 of the Validation Set Approach (VSA):
Model Selection with Hyperparameter Optimization (HPO).
We will optimize and compare the k-Nearest Neighbors Classifier (k-NN) and the Classification Tree model on the diabetes dataset.
    1. We split the data set in 3 subsets: Training Set, Test Set, Validation Set.
    2. We use the Training and Test Sets for
            - training and testing k-NN for different values of k, and then selecting the best k.
            - training and testing the Classification Tree for different values of max_depth, and then selecting the best max_depth.
    3. Finally, we compare the best k-NN model and the best Classification Tree model on the Validation Set to find the best of the best.

To measure prediction performance, we use the Recognition Rate (accuracy).
"""

# Model Selection with HPO mit Validation Set Approach
# from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

################ Load the data
diabetes = pd.read_csv('./Teil Machine Learning/Data/diabetes.csv')

################ Prepare the data
X = diabetes.drop(columns=['Outcome']) # Single out the input
y = diabetes['Outcome'] # Single out the target
y = y.values # Convert the target to a numpy array

################ Train/test/validation split  (60% train, 20% test, 20% val)
# First split off the validation set (20% of the data)
# 20% -> Validation Set / 80% Temp Set
X_temp, X_val, y_temp, y_val = train_test_split(X, y, test_size=0.2, random_state=23, stratify = y)

# Then split the remaining data ("temp") in train (75%) and test (25%)
# von den 80% -> 25% Test Set und 75% Training Set
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=23, stratify = y_temp)
# Note: test_size=0.25 because 0.25 * 0.8 = 0.2, so that test set is 20% of the total data


################ Hyperparameter Optimization for k-NN
k_values = [1, 5, 13, 61, 121, 201, 308]
knn_results = []           # To store the recognition rates for each k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)        # 1. Initialize
    knn_model = knn.fit(X_train, y_train)            # 2. Fit
    R_knn = knn_model.score(X_test, y_test)          # 3. Evaluate
    knn_results.append(R_knn)
    print(f"Test Recognition Rate for k={k} (k-NN): {R_knn * 100:.1f}%")
# Find the best k
best_k_index = knn_results.index(max(knn_results))
best_k = k_values[best_k_index]
print(f"Best k for k-NN: {best_k} with Test Recognition Rate: {knn_results[best_k_index] * 100:.1f}%")



################ Hyperparameter Optimization for Classification Tree
depth_values = [1, 3, 5, 7, 9, 11, 13]
tree_results = []           # To store the recognition rates for each max_depth
for depth in depth_values:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=23)  # 1. Initialize
    tree_model = tree.fit(X_train, y_train)                          # 2. Fit
    R_tree = tree_model.score(X_test, y_test)                        # 3. Evaluate
    tree_results.append(R_tree)
    print(f"Test Recognition Rate for max_depth={depth} (Decision Tree): {R_tree * 100:.1f}%")
# Find the best max_depth
best_depth_index = tree_results.index(max(tree_results))
best_depth = depth_values[best_depth_index]
print(f"Best max_depth for Decision Tree: {best_depth} with Test Recognition Rate: {tree_results[best_depth_index] * 100:.1f}%")


################ Compare the best k-NN and best Decision Tree on the Validation Set
# Best k-NN
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best_model = knn_best.fit(X_train, y_train)
R_knn_val = knn_best_model.score(X_val, y_val)
# Best Decision Tree
tree_best = DecisionTreeClassifier(max_depth=best_depth,
                                   random_state=23)
tree_best_model = tree_best.fit(X_train, y_train)
R_tree_val = tree_best_model.score(X_val, y_val)
# Overall best model
if R_knn_val > R_tree_val:
    print(f"The best model is k-NN with k={best_k} achieving a Validation Recognition Rate of {R_knn_val * 100:.1f}%")
else:
    print(f"The best model is Decision Tree with max_depth={best_depth} achieving a Validation Recognition Rate of {R_tree_val * 100:.1f}%")