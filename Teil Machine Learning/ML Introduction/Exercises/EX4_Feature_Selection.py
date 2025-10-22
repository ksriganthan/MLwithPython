
import sklearn.datasets as ds
import sklearn.feature_selection as fs
import sklearn.neighbors as nb

##### Einen synthetischen Datenset mit 60 Featues generieren
#   Remark:
#       - X ist die Input Space mit 60 Spalten
#       - y ist eine binäre Target Variable (0,1)
X, y = ds.make_classification(n_samples=100, n_features=60, n_redundant=0, n_classes=2)

# Output ist ein ndarray
type(X) # Matrix
type(y) # Vektor oder Skalar

# Shape testen
# y ist in X drin !!
X.shape # (100, 60)


##### Wir benutzen das Univariate Algorithmus mit der Filter-Methode #####

# Filter Methode benutzt eine Quality-Metrix Q, welche abhängig vom Lernalgorithmus ist
# Univariater Ansatz bewertet die Features einzeln mit Q und wählt die k-besten Features mit der höchsten
# Q aus

# Wir benutzen die Klasse 'SelectKBest() fon sklearn Modul 'feature_selection'
# Wir verwenden hier wieder den "initialize-fit-transform" Prozess



# 1. Initialize
# Eine Instanz von SelectKBest() erstellen und folgende Parameter definieren:
# score_func = fs.f_classif (Quality-Metrik definieren)
# 'fs.f_classif' ist der ANOVA F-value. Es misst die Stärke der Beziehung zwischen dem Feature und dem Target
# k = 15 (von 60 auf 15 reduzieren)
kb = fs.SelectKBest(score_func=fs.f_classif, k = 15)
# ANOVA F-Value misst pro Spalte(Feature) Durschnitt je Klassen. Falls die Mittelwerte sicher stark unterscheiden,
# heisst es, dass dieses Feature die Klassen gut trennt. Es berechnet (Unterschied der Mittelwerte / Varianz der Klassen)
# Unterscheidet sich das Feature signifikant zwischen den Zielklassen? - Also wie stark unterscheiden sich der Mittelwert
# der beiden Klassen innerhalb eines Features

# 2. Fit
# Hier lernt der Algorithmus, welche Features gut mit y siginifikant übereinstimmen anhand dem F-Value (Q-Metrik)
# Die Wichtigkeit (Scores) zu jedem Feature wird im kb gespeichert
kb.fit(X, y)
kb.scores_ # Die berechneten Scores zur Q-Metrik - Je höher der F-Wert, desto stärker die Beziehung (Ein Wert pro Feature)

# 3. Transform
# Hier wird nur X gebraucht, um die k-besten Features auszuwählen
X_kb = kb.transform(X)
X_kb.shape # (100, 15)





##### Wir verwenden den iterativen Ansatz mit der Wrapper-Methode #####

# Die Wrapper-Methode benutzt die Performance von einem ML-Algorithmus als ein Wrapper für Q
# In diesem Beispiel verwenden wir KNeighborsClassifier als ein Wrapper
# Wir verwenden die Recognition-Rate als den Performance-Measure für den Wrapper
# Der iterative Ansatz ist kluger als der univariate Ansatz
# Es sucht in 'Space' von den Feature-Subsets für den besten Subset
# Wir verwenden die Sequential Foreward Search (sfs) as a search strategy

# Wrapper: Benutzt die Klasse 'KNeighborsClassifier()' von sklearn Modul 'neighbors'
# Search: Benutzt die Klasse 'SequentialFeatureSelector()' von sklearn Modul 'feature_selection'

# Auch hier verwenden wir den "initialize-fit-transform" Prozess



# 1. Initialize

# Initialize the wrapper
# Erstelle eine Instanz von KNeighborsClassifier-Klasse und spezifiere folgende Parameter:
# n_neighbors = 1 (wir verwenden 1 Nachbar für die Klassifkation)
knn = nb.KNeighborsClassifier(n_neighbors=1)
# In den Trainingsset entscheidet der Algorithmus anhand 1 Nachbar (also Mehrheit gewinnt)
# welche Klasse die aktuelle Zeile angehört -> sehr nervös und kann die Recognition-Rate verschlechtern
# KNN ist ein Lazy-Learner - er behält alle Trainingsdaten


# Initialize the search algorithm
# Erstelle eine Instanz von SequentialFeatureSelector-Klasse und spezifiziere folgende Parameter:
# estimator = knn (wir verwenden knn als Wrapper)
# direction = 'forward (wählt Sequential _foreward_ search)
# n_features_to_select = 15
sfs = fs.SequentialFeatureSelector(estimator=knn, direction='forward', n_features_to_select=15)

# 2. Fit
# Die von sfs generierte Feature Sets werden anhand Performance von knn klassifier bewertet
# Die besten 15 Features werden ausgewählt
sfs.fit(X,y)


# 3. Transform
# Dataset X auf 15 Features runterbrechen
X_sfs = sfs.transform(X)
X_sfs.shape #(100,15)


