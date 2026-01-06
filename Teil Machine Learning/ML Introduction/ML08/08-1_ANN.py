"""
08-1_ANN.py
    - In this module, we will train an Artificial Neural Network (ANN) using the MLPClassifier() from sklearn on the diabetes dataset.
    - We will also discuss how to handle unbalanced datasets and evaluate the model's performance using various metrics.
"""

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ========= DATA PREPARATION =========

# We import the diabetes (originally from Kaggle) using pandas
diabetes = pd.read_csv('./Teil Machine Learning/Data/diabetes.csv')

# Separate input features from target variable
X = diabetes.drop(columns=['Outcome'])
y = diabetes['Outcome'].values

# Split the data to training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.4, random_state=23)

# It is important to scale the data before using neural networks -> WICHTIG: Daten standardisieren/normalisieren vor ANN!!
# -> Sonst haben gewissen Inputs eine höhere Gewichtung
# -> Eher Standardisierung als Normalisierung
#   - StandardScaler() standardizes the features. I.e., it scales each feature to have zero mean and unit variance.
scaler = StandardScaler()

# On the training set, we do the fit and the transform step:
#   - fit: compute the mean and std to be used for later scaling -> mean udn std werden gelernt
#   - transform: perform tha actual standardization by centering and scaling -> die Formel auf alle Daten anwenden
# You can do each step separately (as we did before) or use fit_transform() to combine both steps:
X_train = scaler.fit_transform(X_train)

# On the test set, we only apply the transformation: -> Todo: Gwen fragen
# We use the mean and standard deviation that we learned from the training set:
X_test = scaler.transform(X_test)


# ========= TRAINING A NEURAL NETWORK CLASSIFIER =========

# We use the neural network classifier MLPClassifier.
#   - MLP stands for "Multi Layer Perceptron", which is a common type of *feedforward artificial* neural network (ANN).
#   - The MLPClassifier implements a multi-layer perceptron (MLP) algorithm that trains using *backpropagation*.
#   - It can be used for both *classification and regression tasks*, but here we focus on classification.
#   - It supports *multiple hidden layers*, various activation functions, and different optimization algorithms.
# We apply the initialize-fit-predict-evaluate pattern:

mlp = MLPClassifier(verbose=True) # initialize -> Zeigt den Loss an (z.B. Echte Klasse 1 / Modell sagt 0.99 -> sehr kleiner Loss
mlp.fit(X_train, y_train) # fit
predictions = mlp.predict(X_test) # predict -> Iteration = Epochen
print(predictions)
print("Accuracy:", mlp.score(X_test, y_test)) # evaluate (Recognition Rate) -> 78.89%

# Setting verbose=True
#   - This allows you to see the outputs of the loss function during training.
#   - The loss function measures how well the model is performing during training.
#   - You should see the loss decreasing over epochs, indicating that the model is learning.
#     If the loss does not decrease, it may indicate issues such as inappropriate learning rate or insufficient model complexity.
#   - For classification tasks the log-loss function (logarithmic loss, Cross-Entropy Loss) is used:
#     It quantifies the difference between the predicted probabilities and the actual class labels.
# Cross Entropy: misst, wie gut die vorhergesagten Wahrscheinlichkeit zur echten Klasse passt

mlp = MLPClassifier(verbose=True) # initialize
mlp.fit(X_train, y_train) # fit
print("Accuracy:", mlp.score(X_train, y_train)) # evaluate (Recognition Rate) -> 77.59%
print("Accuracy:", mlp.score(X_test, y_test)) # evaluate (Recognition Rate) -> 82.17%

# What do we observe?
#   - The loss decreases over epochs, indicating that the model is learning.
#   - The iterations stop early, because the improvement in the loss function is smaller than
#     the default tolerance level of tol=0.000100 in 10 consecutive epochs.
# Wenn die Verbesserung 10 mal hintereinander kleiner ist als 0.0001 pro Epoche -> Training stoppen


# Changing the Hyper-Parameters
#   - So far we have instantiated the classifier without hyper-parameters!
#   - The performance of MLPs can be sensitive to hyperparameters like the number of layers, number of neurons
#     and choice of activation functions. Thus proper fine tuning is crucial!
#   The default values for some important hyper-parameters are as follows:
#       - learning_rate_init=0.01
#                ... learning rate for weight updates during training.
#       - activation='relu'
#                ... activation function for the hidden layers.
#               Other possible values are 'identity', 'logistic', 'tanh'.
#       - max_iter=200
#                ... maximum number of epochs for training.
#       - hidden_layer_sizes=100
#                ... defines the architecture of the network
#                By default, the architecture of the network is defined with *one hidden layer with 100 neurons*.
#                This can be changed, for example, as follows:
#                   - hidden_layer_sizes=(10)       ... one hidden layer with 10 neurons.
#                   - hidden_layer_sizes=(30, 20)   ... two hidden layers with 30 and 20 neurons.
#       - random_state=2
#                ... set a seed to make your results reproducible.
#       -> Am Anfang bekommt ANN random Weight -> mit random_state -> jedes mal der gleiche Weight-Vector

# You can now experiment with different hyper-parameters of the MLPClassifier to see how they affect the performance.
#       - For example, you can try changing the number of hidden layers and neurons:
mlp = MLPClassifier(hidden_layer_sizes=(30, 20), max_iter=100, random_state=2, verbose=True) # initialize with 2 hidden layers (30 and 20 neurons)
mlp.fit(X_train, y_train) # fit
print("Accuracy:", mlp.score(X_train, y_train)) # evaluate (Recognition Rate) -> 81.73% -> mögliches Overfitting Todo Gwen fragen
print("Accuracy:", mlp.score(X_test, y_test)) # evaluate (Recognition Rate) -> 79.54%
#       - Notice that with more complex architectures, you might need to increase max_iter to ensure convergence
#         (you may get a warning message otherwise).

# You can also experiment with different activation functions and learning rates to see their impact on training.
#       - For example:
mlp = MLPClassifier(activation='tanh', learning_rate_init=0.001, max_iter=500, random_state=2, verbose=True) # initialize with tanh activation
mlp.fit(X_train, y_train) # fit
print("Accuracy:", mlp.score(X_train, y_train)) # evaluate (Recognition Rate) -> 83.26% -> mögliches Overfitting Todo Gwen fragen
print("Accuracy:", mlp.score(X_test, y_test)) # evaluate (Recognition Rate) -> 76.94%
#       - Again, you may need to adjust max_iter for convergence!

# Notice that the performance of the model can vary significantly based on the choice of hyper-parameters.
# Proper hyper-parameter tuning is essential to achieve good performance with neural networks.

# Notice also that our data set is relatively small.
#   - In practice, neural networks often require large amounts of data to perform well.
#   - With small datasets, simpler models (e.g., logistic regression, decision trees) may perform better and be more interpretable as well.



# ========= DEALING WITH UNBALANCED DATA =========
# How good is our model really?

# Our diabetes dataset has many more negative (no diabetes) than positive (diabetes) cases: -> Mehr no diabetes als diabetes
counts = diabetes['Outcome'].value_counts()
print(counts)
#   - Class 0 (no diabetes): 500 instances
#   - Class 1 (diabetes): 268 instances

# We call this an UNBALANCED DATA SET.
#   - This means that if our model simply predicted "no diabetes" for all instances,
#     it would achieve an accuracy of 500 / (500 + 268) = 65.1% on the entire dataset.
#   - Therefore, a Test Accuracy of around 70% is actually not very impressive!
#   - Since the Accuracy measures the performance in both classes equally, it is not very informative
#     for unbalanced datasets. Other evaluation metrics such as precision, recall, and F1-score are
#     more informative to assess the model's performance.

# Für Unbalanced Data ist es nicht sinnvoll, die Performance in beiden Klassen gleich zu messen
# -> Precision, Recall und F1-score wären sinnvollere Optionen!

# REMEMBER FROM THE BUSINESS INTELLIGENCE LECTURE:
#   - Precision: How many of the predicted positives are actually positive (applies to 2-class problems such as diabetes). -> TP / (TP + FP)
#   - Recall: How many of the actual positives are correctly predicted. -> TP / (TP/FN)
#   - F1-Score: Harmonic mean of precision and recall. -> F1 = 2 * (Precision * Recall) / (Precision * Recall)
# These metrics are especially useful when dealing with imbalanced datasets,
# where one class is significantly more frequent than the other.
# We can use the function classification_report() from sklearn to compute these metrics:

class_report = classification_report(y_test, predictions)
print("Classification Report:\n", class_report) #Man gibt die Lösung von Testset und die Predictions

# What do we observe?
#   - The report provides metrics for each class (0: no diabetes, 1: diabetes).
#   Support on the test set:
#       - Overall: 308 instances
#       - Class 0 (no diabetes): 201 instances
#       - Class 1 (diabetes): 107 instances
#   Accuracy:
#   - Overall Accuracy of the model on the test set: around 70%.
#     This means that the model correctly predicts the outcome for 70% of the 308 test instances, but this counts both classes equally.
#   Other metrics:
#   - The Precision for class 0 (no diabetes) is higher than for class 1 (diabetes).
#     This indicates that the model is better at predicting negative cases.
#   - The recall for class 0 (no diabetes) is also higher than for class 1 (diabetes).
#     This indicates that the model misses some positive cases.
#   - The F1-score for class 0 (no diabetes) is also higher than for class 1 (diabetes).
#     This indicates that the model's performance is not as strong for predicting positive cases.
# Overall, the model seems to perform reasonably well if we look at Accuracy, but the other metrics reveal that
# it struggles with predicting the positive class (diabetes).
# This is a common issue when dealing with unbalanced datasets.
# To improve the model's performance on the minority class (diabetes),
# several techniques exist (but we do not discuss them), e.g.,
#   - resampling
#   - adjusting class weights
#   - using different evaluation metrics during training.


