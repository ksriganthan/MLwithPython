"""
3-1_Decision_Tree.py
"""


from matplotlib import pyplot as plt
from sklearn import datasets as ds
from sklearn import model_selection as ms
from sklearn import tree



#### Load the data
#   We use the integrated iris data set from sklearn.datasets
iris = ds.load_iris() #Bunch
X = iris.data
y = iris.target



#### Fit a tree

# Split the data into training and a test set (Validation Set Approach)
#   - To do this use the function train_test_split() from the sklearn module model_selection.
#   - test_size=0.25. This means we set a ratio of 75% train / 25% test.
#   - random_state=23. The function train_test_split() takes random samples from the data set.
#                      If you want to repeat it later in exactly the same way,
#                      you can make the random selection reproducible by setting a seed, e.g., 1. for the pseudo-randomizer.
#                      This number is arbitrary - it just has to be the same in every run.
#   - stratify=y. Remember that y is the target variable that holds the class values.
#                 This option preserves the class distribution when splitting the sample.
#                 That means both, the training and the test set, have the same distribution of class values
#                 as the original sample.
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.25, random_state=23, stratify=y)


# Fit (learn) a tree
#   To do this, we use the function DecisionTreeClassifier() from the module tree of sklearn.
#   It follows the initialize-fit-transform process.
#
#   1. Initialize
#       Create an instance of the DecisionTreeClassifier() class by specifying the following parameters:
#       - criterion='gini'. The 'criterion' parameter specifies the metric to measure the quality of a split.
#       - random_state=0. To obtain a deterministic behaviour during fitting, `random_state` has to be fixed to an integer.
#   DecisionTreeClassifier nicht Regression!
clf = tree.DecisionTreeClassifier(criterion='gini', random_state=0) # Diese Algos haben oft bei Exceptions (z.B. Ties oder Nicht-Pure) Randomizer drin
# Random-State = 0 stellt sicher, dass es beim Bauen des Modells, immer der gleiche Seed für die Vergleichbarkeit verwendet wird
# Kann auch eine andere Zahl sein als 0 -> Hauptsache eine fixe Zahl, die für Reproduzierbarkeit sorgt
#
#
#   2. Fit
#       Build the tree structure based on the training data.
#       The algorithm recursively splits the data at each node. To choose the best split at each node,
#       it evaluates all potential splits and chooses the one that minimizes class impurity based on the gini index.
#
#       Nimmt dann diese eine bestimmte Reihenfolge von Features, die für den grössten Impurity-Decrease
#       (Angefangen bei Root-Node: von jeweiligen Parent-Node bis Child Node) sorgt
#       Lernt hier wie man an sich splittet und z.B. um overfitting zu vermeiden -> wann man aufhört
model = clf.fit(X_train, y_train)
#
#   2. (Transform /) Predict
#       - In the context of classifiers like the DecisionTreeClassifier(), we do not 'transform' anything.
#       -> Wäre nur z.B. bei Binarization oder Discretization
#         Instead of 'transforming' the target variable, we predict it!
#         After prediction, we evaluate how well the predictions match the actual class values.
#         So, in this case, the process is actually 'initialize-fit-predict-evaluate'.
#       - To predict the class values, we use the method .predict().
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
#       Note:
#       - When we let the model predict the class values, we only give to the model the values of the input features,
#         X_train, X_test.
#       - The _actual_ class values we observed are stored in the target variables y_train and y_test. We hold them
#         back, so that we can compare the _predicted_ class values with the _actual_ class values later on.


# Evaluate the learned tree
#       To see how well our tree performs, we compare the predicted values with the actual class values.
#       We can do that using the method .score().
print("Unpruned tree:")
print("Accuracy on training set:",  model.score(X_train, y_train)) # 100%, da das Model auf dieselben Daten trainiert wurden (Overfitting)
print("Accuracy on test set:", model.score(X_test, y_test))
#       WICHTIG:
#       - model.score(X_train, y_train) internally calls the function model.predict(X_train) to produce the predicted class
#         values. (Same for the test set!)
#       - It then compares the predicted class values to the actual class values to calculate the Recognition Rate.
#       - The Recognition Rate is a measure of accuracy of the classifier.


# What does the evaluation results tell us?
#
#       We see that the accuracy on the training set is 100%.
#           - The reason is that we did not prune the tree: we let it grow all the way down without a stop criterion.
#           - The algorithm only stopped the splitting when the leave nodes on the training set where 100% pure (no mixed classes).
#       The accuracy on the test set is smaller (about 98%).
#           - The reason is that the test set contains data objects that the tree did not 'see' while splitting.
#           - Thus, it could not take them into account when
#       This is a strong indication that our tree is OVERFITTING the training data! (See introductory lecture.)

#       Todo: Gwen fragen
#       Training und Test Data kommen aus derselben Verteilung, also sie haben verhältnismässig gleich viele Datensätze je Target Class Label
#       Wahrscheinlichkeit besteht, dass die Regeln sehr gut zu den Test Daten passen -> deshalb so hohe Accuracy bei Test Set

# Visualize the tree
#       1.  To create a new figure use pyplot.figure() from matplotlib.
#           A figure in Matplotlib is a container that holds all the elements of a plot (axes, titles, labels, etc.).
fig = plt.figure(figsize=(10,7))
#       2.  'Populate' the container with the decision tree provided by the learned model.
#           To do that use tree.plot_tree() from sklearn.
#           It is specifically designed to visualize the output of tree.DecisionTreeClassifier()
tree.plot_tree(model,
               feature_names=iris.feature_names,
               class_names=iris.target_names,
               rounded=True, # Rounded node edges
               filled=True, # Adds color according to class
               proportion=False); # Displays the proportions of class samples instead of the whole number of samples
#       3.  Render the populated figure.
#           It opens a window with the visual representation of the figure.
#           If you use an interactive backend for matplotlib, such as 'Qt5Agg', you can interact with the plot.
plt.show()

# What can we see in the plot?
#
#       - In the intermediate nodes, the first line gives the split criterion (e.g., 'petal width (cm) <= 0.8').
#         Notice that the leaf nodes on the bottom don't have a split criterion, since they are not split anymore.
#       - The gini index gives the node purity. The gini index of the leaf nodes is 0.0, which indicates that they are pure.
#       - 'samples = ...' gives the relative sample size after the split.
#       - The vector 'values' contains the percentages/numbers of samples of each class (we have 3 classes here).
#         Notice that, in the leaf nodes, only one element of 'values' is 1.0, the two others are always 0.0.
#       - 'class = ...' tells us the majority class in the node.


#### Prune the tree
#
#   To avoid OVERFITTING of our tree to the training data, we can prune the tree.
#   - To do that, we define a stop criterion for splitting that prevents the tree from 'growing all the way down' until leaf nodes are pure.
#     One possible stop criterion is to set a maximum tree depth: -> Hyperparameter:Prune Threshold; Maximum Tree Depth
#   - max_depth=3 means that max. 3 consecutive questions can be asked:
#

#Level 0 (Root):    [petal width <= 0.8?]        ← Frage 1
#                          /        \
#Level 1:        [Frage 2]         [Frage 2']    ← Frage 2
#                  /    \            /    \
#Level 2:      [F3]    [F3']     [F3'']  [F3'''] ← Frage 3
#                /\      /\        /\       /\
#Level 3:     Blätter (STOPP bei max_depth=3)  -> Wenn hier die einzelnen Subsets nicht rein sind -> Majority Vote
#

# Alternative Stopping-Kriterien:
# min_samples_split (Mindestanzahl Samples für weiteren Split)
# min_samples-leaf (Mindestanzahl Samples pro Blatt)
# max_leaf_nodes: Maximale Anzahl Blätter


#   We follow the steps of the 'initialize-fit-predict-evaluate' process:
#
#       1. Initialize
clf_pruned = tree.DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=3) #Hier als zusätzlicher Parameter
#
#       2. Fit
model_pruned = clf_pruned.fit(X_train, y_train)
#
#       3. Predict
train_pred_pruned = model_pruned.predict(X_train)
test_pred_pruned = model_pruned.predict(X_test)
#
#       4. Evaluate
print("Tree with max_depth=3:")
print("Accuracy on training set:",  model_pruned.score(X_train, y_train))
print("Accuracy on test set:", model_pruned.score(X_test, y_test))
print("")

# Stopping the splitting early has led to a lower training accuracy.
# The test accuracy usually improves (gets higher). -> Normalerweise müsste der Test Accuracy besser sein, als vorhin
# Yet, since our test set is extremely small, the test accuracy can also be lower than before. -> Kann passieren, da Test Set sehr klein ist
# Es ist abhängig davon, was für Datenobjekte und wie viele davon verwendet wurden.
# Wichtig: Die ML-Algorithmus Resultate sind Statistische Repräsentationen und arbeiten mit Durchschnitte und mit der Voraussetzung,
# dass die Samples gross genug sind, um stabil zu sein!

# Visualize the tree
fig = plt.figure(figsize=(8,6))
tree.plot_tree(model_pruned,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          rounded=True, # Rounded node edges
          filled=True, # Adds color according to class
          proportion=False); # Displays the proportions of class samples instead of the whole number of samples
plt.show()

# What can we see in the plot?
#
#       - The first 4 levels of the tree are exactly the same as before.
#       - After 4 levels (3 questions) it stops.
#       - The leaf nodes are not pure now. -> Majority Vote



#### Hyperparameter Optimization: Finding the Sweet Spot

# Remember that we need to find the sweet spot between under- and overfitting:
#   - The flexibility of a model determines if it is underfitting, overfitting, or if it is a good fit.
#   - To find the sweet spot between over- and underfitting, we must do the whole 'initialize-fit-predict-evaluate'
#     cycle several times, while changing model flexibilities. We then choose the model with the highest test accuracy.
#   - For our tree, we can change the flexibility by changing the value of the stopping criterion 'max_depth':
#     The higher 'max_depth' is, the more flexible is the tree. (It can adapt more to the training data.)

# Let's do that:
highest_test_accuracy = 0
# best_depth = 0
for d in range(2, 8, 1): # from 2 to 7 in steps of 1 ->  generates 2, 3, 4, 5, 6, 7
        clf = tree.DecisionTreeClassifier(criterion='gini',
                                         random_state=0,
                                         max_depth=d)
        clf.fit(X_train, y_train)
        train_accuracy = clf.score(X_train, y_train)
        test_accuracy = clf.score(X_test, y_test)
        print("depth=", d)
        print("Train accuracy:", train_accuracy)
        print("Test accuracy:", test_accuracy)
        print("")
        # if we got a better score, store the score and parameters
        if test_accuracy > highest_test_accuracy:
            highest_test_accuracy = test_accuracy
            best_depth = d
            best_model = clf
print("Best tree:")
print("depth=", best_depth)
print("Test accuracy:", highest_test_accuracy)
# Hätte man hier noch das beste Modell genommen und noch auf ein Validation Set geprüft -> Model Evaluation with HPO

# What can we see in the output?
#   - The training accuracy increases with increasing flexibility. -> Desto mehr Splits, desto höher die Accuracy
#   - The test accuracy increases with flexibility, and then stabilizes at depth 4. -> Steigert bis es nur noch stagniert (oder sinkt)
#     This behavior deviates a bit from the typical behavior, where the test accuracy would decrease again at a certain point. -> irgendwann würde er sinken
#     Das ist weil der Iris Datensatz sehr klein sind -> Test und Train sehr klein -> Deshalb sieht man nicht das typisch stabile Kurvenverhalten
#     Das Verhältnis von Training/Dataset und deren Random Choice Verteilung (random_state/stratify) beeinflusst ebenfalls die Test Accuracy-Ergebnisse (nicht-stabil)
#
# Note:
#   - Regarding test accuracy all models from depth=4 on are equally good. -> Alle Modelle ab Depth 4 aufwärts sind gleich gut
#     Yet, we rather choose the simpler model, since it is easier to interpret. -> Wir nehmen einfach das einfachere Modell (weniger Split = weniger Komplex)
#   - In reality, we would do more tests with different random choices in the train/test split. -> Man würde Random Choices auch ändern in der Realität (Zufallsverteilung)!
# -> Also einen anderen Random-State-Wert
#     We will discuss this later under the headline of "cross-validation".


# Visualize the tree
fig = plt.figure(figsize=(10,7))
tree.plot_tree(best_model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          rounded=True,
          filled=True,
          proportion=True);
plt.show()

# Anzahl Tiefe = Anzahl Edges vom Root zum tiefsten Blatt (inkl. Testcondition vom Root-Node)


