"""
06-2_VSA_for_model_selection.py
In this module, we try out the Validation Set Approach (VSA).
    - First, we use it to evaluate the prediction performance of the k-NN classifier.
    - Second, we use it to evaluate the prediction performance of the unweighted k-NN.
    - We compare both models based on their recognition rates on the test set and select the better one.

To measure prediction performance, we use the Test Recognition Rate (test accuracy).
"""



################ Preliminaries

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



################ Load the data

#       We import the diabetes dataset again:
diabetes = pd.read_csv('./Teil Machine Learning/Data/diabetes.csv')  # Use this line if you run the code cell by cell.
# diabetes = pd.read_csv('../Data/diabetes.csv')    # Use this line if you run the whole module using the Run button

#       Since we want to use the KNeighborsClassifier again we must separate input and target:
X = diabetes.drop(columns=['Outcome'])
y = diabetes['Outcome']
#       The target must be provided as a numpy array:
y = y.values


################ Preprocessing for the Validation Set Approach (VSA)

##  Data preprocessing: train/test split
#  - We want to use the Validation Set Approach (VSA) for evaluation later.
#  - Thus, we must split our sample in training and test set before we fit the model to the training data.
#  - To do that, we use the function train_test_split() from sklearn.model_selection.
#   It requires the following parameters:
#      - X: our input data
#      - y: our target variable
#      - test_size=0.4 specifies that the split ratio for train/test set is 60/40.
#      - random_state=23 sets the seed for the pseudo-randomizer to 23 (arbitrary, but fixed number). -> Immer der gleiche Startwert
#        This ensures that we always get the same train/test split when we run the code multiple times. -> Wichtig für Vergleiche
#      - stratify = y ensures that both, training and test set, have the same class distribution as the whole sample.
#        Stellt sicher, dass die gleiche prozentuale Anteil der verschiedenen Labels gleich sind in Training sowie Test Data
# The function returns four objects:
#      - X_train: input features of the training set
#      - X_test: input features of the test set
#      - y_train: target variable of the training set
#      - y_test: target variable of the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23, stratify = y)

#   Double-check the data types of the train/test splits:
type(X_train) # pandas.core.frame.DataFrame
type(y_train) # numpy.ndarray
type(X_test)  # pandas.core.frame.DataFrame
type(y_test)  # numpy.ndarray
#   Double-check the shapes of the train/test splits:
X_train.shape # (460, 8)
X_test.shape  # (308, 8)  # Our test data contains 308 patients (rows)
y_train.shape # (460,)
y_test.shape  # (308,)    # The target variable for the test data also contains 308 entries - one per patient.



################ "Learn the k-NN model from the TRAINING data"
# (Aka: "Fit the k-NN algorithm to the TRAINING data")
#
#   1. Initialize
knn = KNeighborsClassifier(n_neighbors=9)
#
#   2. Fit
#      Now we fit the kNN algorithm ONLY to the TRAINING data (X_train, y_train)!
#      Remember that the fit step of k-NN does not really fit a model, but only stores the training data
#      and the decision rule in the model object. This is because k-NN is a lazy learner.
knn_model = knn.fit(X_train, y_train)
#
#   3. Predict
#      Now we use the k-NN decision rule and the TRAINING data stored in the model object knn_model
#      to make predictions for the patients in the TEST data (X_test).
#      This means we feed the input values of all patients in the test set into the k-NN decision rule
#      and let it predict whether each patient in this test set has diabetes or not:
y_pred = knn_model.predict(X_test)

# Looking at the predictions for the test data:
print(y_pred)
y_pred.shape # (308,) - one prediction per patient in the test set.



################ Evaluate the prediction performance of the k-NN Classifier using VSA

#   To evaluate the prediction performance of the k-NN Classifier on the test data,
#   we compare the predicted target values y_pred with the true target values y_test.
#   To do that, we can use the method score() from the model object knn_model.
#   It internally preforms 3 steps:
#       1. Use the model stored in the model object knn_model to predict the target values of the test data X_test.
#           ->knn_model.predict(X_test) -> Array mit allen predicted Klassen pro Zeile
#       2. Compare the predictions with the true values y_test.
#           -> vergleich = (y_pred_internal == y_test)
#       3. From the comparison, calculate the Recognition Rate.
#           -> Anzahl Richtige durch Gesamtzeilen
R = knn_model.score(X_test, y_test)
#   The Recognition Rate on the test data is 73.7%
#   This means that the k-NN Classifier correctly predicts whether a patient has diabetes or not
#   for 73.7% of all patients in the test data set.
print(f"Recognition Rate on the test set for k={knn_model.n_neighbors}: {R * 100:.1f}%")

#   The Recognition Rate on the training data is 78.9%
#   Das Model generalisiert gut, da der Abstand nicht gross ist (sonst wäre es Overfitting)
RT = knn_model.score(X_train,y_train)
print(f"Recognition Rate on the training set for k = {knn_model.n_neighbors}: {RT*100:.1f}%")



################ Learn and evaluate the WEIGHTED k-NN model

# The only difference to the unweighted k-NN model above is in the initialization step:
# Here, we add the parameter weights='distance'.
# This means that the neighbours are weighed according to their distance to the new object.
# All other steps remain the same:
knn_w = KNeighborsClassifier(n_neighbors=9, weights='distance')   # 1. Initialize
knn_model_w = knn_w.fit(X_train, y_train)                         # 2. Fit
y_pred_w = knn_model_w.predict(X_test)                            # 3. Predict
R_w = knn_model_w.score(X_test, y_test)                           # 4. Evaluate
print(f"Test Recognition Rate in test set for k={knn_model_w.n_neighbors} (weighted): {R_w * 100:.1f}%") # 74.0%

R_wT = knn_model_w.score(X_train, y_train)
print(f"Test Recognition Rate in training set for k={knn_model_w.n_neighbors} (weighted): {R_wT * 100:.1f}%") # 100.0% -> Overfitting
# Which model performs better on the test set - the unweighted or the weighted k-NN Classifier?
# Bei Test Set ist Weighted leicht besser, jedoch im Training Set ist Accuracy bei Weighted bei 100%. Es gibt ein sehr grosser Gap
# Zwischen Training und Set, deshalb kann man nicht sage, dass Weighted besser ist und dazukommt noch, dass wir nicht wissen welche
# k die Beste ist
# Ausserdem generalisiert Unweighted Neighbour bei k = 9 besser als Weighted Neighbour
# Todo Mit Gwen anschauen







