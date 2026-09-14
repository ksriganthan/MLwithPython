# Machine Learning with Python (Deutsch)

🇬🇧 **English version:** [Machine Learning with Python (English)](#machine-learning-with-python-english)

Ein umfassendes Repository zum strukturierten Erlernen von **Python-Grundlagen** und **Machine-Learning-Methoden** im Rahmen des Moduls *Business Analytics / Machine Learning with Python* (FHNW).

Der Fokus liegt auf **Verständnis, sauberer Umsetzung und praktischer Anwendung** zentraler ML-Konzepte – nicht auf Black-Box-Nutzung von Frameworks.

---

## Inhaltsverzeichnis

- [Über das Projekt](#über-das-projekt)
- [Python-Grundlagen](#python-grundlagen)
- [Machine Learning](#machine-learning)
  - [Supervised Learning](#supervised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
  - [Model Evaluation](#model-evaluation)
- [Projektstruktur](#projektstruktur)
- [Installation & Verwendung](#installation--verwendung)
- [Verwendete Bibliotheken](#verwendete-bibliotheken)
- [Datensätze](#datensätze)
- [Lernziele](#lernziele)

---

## Über das Projekt

Dieses Repository dokumentiert meine Lernreise durch Python und Machine Learning. Es enthält praktische Implementierungen von ML-Algorithmen, Datenanalyse-Techniken und Python-Programmierübungen. Alle Konzepte werden nachvollzogen und bewusst angewendet: Verfahren, Hyperparameter und Bewertungsmethoden werden erklärt, statt Bibliotheksfunktionen als Black Box zu benutzen.

### Warum Python für Machine Learning?

Python ist die führende Sprache im Data-Science- und ML-Bereich. Die Gründe:
- **Einfacher Syntax**: Leicht zu erlernen und zu lesen
- **Umfangreiche Bibliotheken**: NumPy, pandas, scikit-learn, matplotlib
- **Grosse Community**: Viele Ressourcen und Support
- **Vielseitigkeit**: Von Datenanalyse bis Produktionsumgebungen

---

## Python-Grundlagen

### Teil Python - Wochenweise Organisation

Das Repository enthält Python-Übungen und -Konzepte, organisiert nach Lernwochen:

#### **Woche 01**
- Erste Schritte mit Python
- Datentypen und Variablen
- Mathematische Operationen mit `sympy`
- Primzahlenprüfung

#### **Woche 03**
- **Datenstrukturen**: Listen, Tupel, Sets, Dictionaries
- **Dictionaries**: Schlüssel-Wert-Paare, JSON-Konvertierung
- **Tupel**: Unveränderliche Datenstrukturen (Anwendungsfälle: GPS-Koordinaten, RGB-Werte)
- **Sets**: Mengenoperationen, Sortierung

#### **Übungen**
Umfangreiche Übungen zu:
- **Chapter 6-9**: File I/O, Datenmanipulation
- **matplotlib**: Datenvisualisierung (Plots, Scatter-Plots, Histogramme)
- **JSON-Verarbeitung**: 
  - `json.dumps()`: Python Dictionary → JSON String
  - `json.loads()`: JSON String → Python Dictionary
  - Error Handling bei JSON-Parsing

#### **Exam Preparation**
- Praxisnahe Übungsaufgaben
- Testdaten und Beispielimplementierungen

### Wichtige Python-Konzepte

```python
# Dictionary: ordered, key-value pairs
person = dict(name="Alice", age=30, city="New York")

# Set: unordered, unique elements
namen = {"Kapi", "Sushana", "Karim", "Loic", "Nuria"}
sorted_list = sorted(namen)

# Tuple: immutable, für unveränderliche Daten
gps_coordinates = (47.5596, 7.5886)  # Basel
rgb_color = (255, 128, 0)  # Orange
```

---

## Machine Learning

### Supervised Learning

#### 1. **k-Nearest Neighbors (k-NN)**
   
**Konzept**: Lazy Learner - speichert Trainingsdaten und klassifiziert neue Punkte basierend auf den k nächsten Nachbarn.

**Implementierung**:
- `06-1_kNN.py`: Unweighted und Weighted k-NN
- Diabetes-Datensatz: Vorhersage ob Patient Diabetes hat
- Hyperparameter: k (Anzahl Nachbarn)

```python
from sklearn.neighbors import KNeighborsClassifier

# Initialize-Fit-Predict Process
knn = KNeighborsClassifier(n_neighbors=9)  # Initialize
knn_model = knn.fit(X_train, y_train)      # Fit (speichert nur Daten)
predictions = knn_model.predict(X_test)    # Predict

# Weighted k-NN: Näherliegende Nachbarn haben mehr Einfluss
knn_weighted = KNeighborsClassifier(n_neighbors=9, weights='distance')
```

**Key Insights**:
- **Lazy Learner**: Keine explizite Trainingsphase - speichert nur Daten
- **Odd k**: Ungerade k-Werte vermeiden Unentschieden bei Mehrheitsentscheidung
- **Distance Weighting**: Näherliegende Nachbarn können stärker gewichtet werden

---

#### 2. **Decision Trees (Entscheidungsbäume)**

**Konzept**: Eager Learner - lernt Baumstruktur basierend auf Gini-Index zur Minimierung der Klassenunreinheit.

**Implementierung**:
- `07-1_Decision_Tree.py`: Vollständiger und geprunter Baum
- Iris-Datensatz: Klassifikation von Blumenarten
- Pruning mit `max_depth` zur Overfitting-Vermeidung

```python
from sklearn import tree

# Unpruned Tree (kann zu Overfitting führen)
clf = tree.DecisionTreeClassifier(criterion='gini', random_state=0)
model = clf.fit(X_train, y_train)

# Pruned Tree (Pre-Pruning)
clf_pruned = tree.DecisionTreeClassifier(criterion='gini', 
                                         random_state=0, 
                                         max_depth=3)
model_pruned = clf_pruned.fit(X_train, y_train)

# Visualisierung
tree.plot_tree(model_pruned, 
               feature_names=iris.feature_names,
               class_names=iris.target_names,
               rounded=True, 
               filled=True)
```

**Key Insights**:
- **Gini-Index**: Misst Klassenunreinheit an jedem Knoten
- **Pre-Pruning**: `max_depth` stoppt Wachstum frühzeitig
- **Interpretierbarkeit**: Einfachere Modelle sind besser interpretierbar

---

#### 3. **Artificial Neural Networks (ANN)**

**Konzept**: Multi-Layer Perceptron mit Hidden Layers für komplexe Mustererkennung.

**Implementierung**:
- `08-1_ANN.py`: Neuronale Netze mit verschiedenen Architekturen
- Konfigurierbare Hidden Layers und Neuronen

```python
from sklearn.neural_network import MLPClassifier

# 2 Hidden Layers mit 30 und 20 Neuronen
mlp = MLPClassifier(hidden_layer_sizes=(30, 20), 
                    max_iter=100, 
                    random_state=2, 
                    verbose=True)
mlp.fit(X_train, y_train)
accuracy = mlp.score(X_test, y_test)
```

**Key Insights**:
- **Activation Functions**: Verschiedene Funktionen beeinflussen das Training
- **Learning Rate**: Anpassung der Lerngeschwindigkeit
- **Convergence**: `max_iter` muss ggf. erhöht werden

---

### Unsupervised Learning

#### 4. **Hierarchical Clustering**

**Konzept**: Agglomeratives Clustering - baut Hierarchie von Clustern bottom-up auf.

**Implementierung**:
- `09-1_HierClust.py`: Agglomerative Clustering mit Complete Linkage
- Synthetic Dataset mit 4 Blobs
- Dendrogramm-Visualisierung

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, complete

# Clustering
agg = AgglomerativeClustering(linkage="complete", n_clusters=4)
assignment = agg.fit_predict(X)

# Hyperparameter Tuning mit Silhouette Score
for k in range(2, 8):
    agg = AgglomerativeClustering(linkage="complete", n_clusters=k)
    assignment = agg.fit_predict(X)
    score = silhouette_score(X, assignment)
    print(f"K = {k}, Silhouette Score = {score}")

# Dendrogramm (mit scipy)
linkage_array = complete(X)
dendrogram(linkage_array, color_threshold=10)
```

**Key Insights**:
- **Complete Linkage**: Maximaler Abstand zwischen Clustern
- **Silhouette Score**: Evaluationsmetrik für Clustering-Qualität
- **Dendrogramm**: Visualisiert hierarchische Struktur
- **Fit-Predict**: Kombiniert Fitting und Zuordnung in einem Schritt

---

### Model Evaluation

#### **Validation Set Approach (VSA)**

**Konzept**: Split in Train/Test für Modellauswahl.

**Implementierung**:
- `06-2_VSA_for_model_selection.py`: k-NN Modellauswahl
- `07-2_VSA_ApplicationCase4.py`: k-NN vs. Decision Tree Vergleich

```python
from sklearn.model_selection import train_test_split

# Train/Test Split (60% / 40%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=23, stratify=y
)

# Train/Test/Validation Split (60% / 20% / 20%)
X_temp, X_val, y_temp, y_val = train_test_split(
    X, y, test_size=0.2, random_state=23, stratify=y
)
X_train, X_test, y_train, y_test = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=23, stratify=y_temp
)
```

---

#### **Cross-Validation (CV)**

**Konzept**: k-Fold Cross-Validation für robustere Modellbewertung.

**4 Anwendungsfälle**:

1. **Application Case 1**: Model Evaluation ohne HPO
   - `07-3_CV_ApplicationCase1.py`
   - 9-NN auf gesamtem Datensatz evaluieren

2. **Application Case 2**: Model Selection ohne HPO
   - `07-4_CV_ApplicationCase2.py`
   - 9-NN vs. Decision Tree (max_depth=5) vergleichen

3. **Application Case 3**: Model Evaluation mit HPO
   - `07-5_CV_ApplicationCase3.py`
   - Optimales k für k-NN finden mit CV

4. **Application Case 4**: Model Selection mit HPO
   - `07-6_CV_ApplicationCase4.py`
   - k-NN und Decision Tree optimieren und vergleichen

```python
from sklearn.model_selection import cross_val_score
import numpy as np

# 5-Fold Cross-Validation
knn = KNeighborsClassifier(n_neighbors=9)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
mean_score = np.mean(scores)

# Hyperparameter Optimization mit CV
k_values = [1, 5, 13, 61, 121, 201, 307]  # nur ungerade Werte, damit keine Stimmengleichheit entsteht
cv_results = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_results.append(np.mean(scores))

best_k = k_values[cv_results.index(max(cv_results))]
```

**Key Insights**:
- **Robustness**: CV liefert stabilere Schätzungen als einzelner Train/Test Split
- **No Data Waste**: Jeder Datenpunkt wird für Training und Testing verwendet
- **HPO**: CV ist essentiell für Hyperparameter-Tuning

---

## Projektstruktur

```
MLwithPython/
├── Teil Python/
│   ├── Woche01/              # Python Basics
│   ├── Woche03/              # Datenstrukturen
│   ├── Übungen/              # Praktische Übungen
│   │   ├── Chapter6-9/       # File I/O, Datenmanipulation
│   │   └── matplotlib/       # Datenvisualisierung, u. a. plt_examples/plt_examples.ipynb
│   └── examPreparation/      # Prüfungsvorbereitung
│
├── Teil Machine Learning/
│   ├── Data/                 # Datensätze (diabetes.csv, etc.)
│   └── ML Introduction/
│       ├── ML06 kNN and Evaluation/
│       │   ├── 06-1_kNN.py                        # k-NN Implementierung
│       │   └── 06-2_VSA_for_model_selection.py   # Validation Set Approach
│       ├── ML07 Trees and CV/
│       │   ├── 07-1_Decision_Tree.py             # Decision Trees
│       │   ├── 07-2_VSA_ApplicationCase4.py      # VSA Anwendung
│       │   ├── 07-3_CV_ApplicationCase1.py       # CV: Evaluation ohne HPO
│       │   ├── 07-4_CV_ApplicationCase2.py       # CV: Selection ohne HPO
│       │   ├── 07-5_CV_ApplicationCase3.py       # CV: Evaluation mit HPO
│       │   └── 07-6_CV_ApplicationCase4.py       # CV: Selection mit HPO
│       ├── ML08/
│       │   └── 08-1_ANN.py                       # Neuronale Netze
│       └── ML09/
│           └── 09-1_HierClust.py                 # Hierarchical Clustering
│
├── requirements.txt          # Python Dependencies
└── README.md                # Diese Datei
```

---

## Installation & Verwendung

### Voraussetzungen

- Python 3.10 oder neuer
- pip (Python Package Manager)

### Installation

```bash
# Repository klonen
git clone https://github.com/ksriganthan/MLwithPython.git
cd MLwithPython

# Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate  # Auf Windows: venv\Scripts\activate

# Dependencies installieren
pip install -r requirements.txt
```

### Verwendung

```bash
# Einzelne Scripts ausführen
python "Teil Machine Learning/ML Introduction/ML06 kNN and Evaluation/06-1_kNN.py"

# Jupyter Notebooks
jupyter notebook "Teil Python/Übungen/matplotlib/plt_examples/plt_examples.ipynb"
```

---

## Verwendete Bibliotheken

### Core ML & Data Science
- **pandas** (2.x): Datenmanipulation und -analyse
- **numpy** (1.x): Numerische Berechnungen
- **scikit-learn** (1.x): Machine Learning Algorithmen
- **matplotlib** (3.x): Datenvisualisierung
- **scipy** (1.x): Wissenschaftliche Berechnungen (Clustering, Statistik)

### Utility
- **sympy** (1.14.0): Symbolische Mathematik
- **deep-translator**: Sprachübersetzungen

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib scipy sympy deep-translator
```

---

## Datensätze

### 1. **Diabetes Dataset**
- **Quelle**: `Teil Machine Learning/Data/diabetes.csv`
- **Verwendung**: k-NN, Decision Trees, Cross-Validation
- **Features**: 8 medizinische Merkmale (Glucose, BMI, Age, etc.)
- **Target**: Outcome (0 = kein Diabetes, 1 = Diabetes)
- **Samples**: 768 Patienten

### 2. **Iris Dataset**
- **Quelle**: `sklearn.datasets.load_iris()`
- **Verwendung**: Decision Trees
- **Features**: 4 Blütenmerkmale
- **Target**: 3 Blumenarten (Setosa, Versicolor, Virginica)
- **Samples**: 150 Blumen

### 3. **Synthetic Clustering Dataset**
- **Quelle**: `sklearn.datasets.make_blobs()`
- **Verwendung**: Hierarchical Clustering
- **Features**: 2D-Koordinaten
- **Clusters**: 4 Blobs
- **Samples**: 1000 Datenpunkte

---

## Lernziele

### Python-Kenntnisse
✅ Datentypen und Datenstrukturen (Lists, Tuples, Sets, Dictionaries)  
✅ Kontrollstrukturen (Loops, Conditionals)  
✅ Funktionen und Modularisierung  
✅ File I/O und JSON-Verarbeitung  
✅ Datenvisualisierung mit matplotlib  

### Machine Learning Konzepte
✅ **Supervised Learning**: k-NN, Decision Trees, Neural Networks  
✅ **Unsupervised Learning**: Hierarchical Clustering  
✅ **Model Evaluation**: Train/Test Split, Cross-Validation  
✅ **Hyperparameter Optimization**: manuelle Parametersuche über k-Fold-CV  
✅ **Overfitting**: Erkennung und Vermeidung (Pruning, Validation)  
✅ **Bias-Variance Tradeoff**: am Vergleich von ungeprunetem und gepruntem Baum gezeigt  

### Praktische Fähigkeiten
✅ Scikit-learn Workflow: Initialize-Fit-Predict  
✅ Datenaufbereitung mit pandas  
✅ Modellbewertung mit Metriken (Accuracy, Silhouette Score)  
✅ Visualisierung von ML-Ergebnissen  

---


## Autor

**Kapischan Sriganthan**  

---

## Lizenz

Dieses Repository dient ausschliesslich Lernzwecken im Rahmen meines Studiums an der FHNW. Es steht unter keiner Open-Source-Lizenz.

---


## Machine Learning with Python (English)

A comprehensive repository for structured learning of **Python fundamentals** and **Machine Learning methods** as part of the *Business Analytics / Machine Learning with Python* course (FHNW).

The focus is on **understanding, clean implementation, and practical application** of core ML concepts – not on black-box usage of frameworks.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Python Fundamentals](#python-fundamentals)
- [Machine Learning](#machine-learning-1)
  - [Supervised Learning](#supervised-learning-1)
  - [Unsupervised Learning](#unsupervised-learning-1)
  - [Model Evaluation](#model-evaluation-1)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Used Libraries](#used-libraries)
- [Datasets](#datasets)
- [Learning Objectives](#learning-objectives)

---

## About the Project

This repository documents my learning journey through Python and Machine Learning. It contains practical implementations of ML algorithms, data analysis techniques, and Python programming exercises. All concepts are worked through and applied deliberately: algorithms, hyperparameters and evaluation methods are explained rather than used as a black box.

### Why Python for Machine Learning?

Python is the leading language in the Data Science and ML field due to:
- **Simple Syntax**: Easy to learn and read
- **Extensive Libraries**: NumPy, pandas, scikit-learn, matplotlib
- **Large Community**: Many resources and support
- **Versatility**: From data analysis to production environments

---

## Python Fundamentals

### Python Part - Weekly Organization

The repository contains Python exercises and concepts organized by learning weeks:

#### **Week 01**
- First steps with Python
- Data types and variables
- Mathematical operations with `sympy`
- Prime number checking

#### **Week 03**
- **Data Structures**: Lists, Tuples, Sets, Dictionaries
- **Dictionaries**: Key-value pairs, JSON conversion
- **Tuples**: Immutable data structures (use cases: GPS coordinates, RGB values)
- **Sets**: Set operations, sorting

#### **Exercises**
Extensive exercises on:
- **Chapter 6-9**: File I/O, data manipulation
- **matplotlib**: Data visualization (plots, scatter plots, histograms)
- **JSON Processing**: 
  - `json.dumps()`: Python Dictionary → JSON String
  - `json.loads()`: JSON String → Python Dictionary
  - Error Handling in JSON parsing

#### **Exam Preparation**
- Practical exercise tasks
- Test data and example implementations

### Important Python Concepts

```python
# Dictionary: ordered, key-value pairs
person = dict(name="Alice", age=30, city="New York")

# Set: unordered, unique elements
namen = {"Kapi", "Sushana", "Karim", "Loic", "Nuria"}
sorted_list = sorted(namen)

# Tuple: immutable, for unchangeable data
gps_coordinates = (47.5596, 7.5886)  # Basel
rgb_color = (255, 128, 0)  # Orange
```

---

## Machine Learning

### Supervised Learning

#### 1. **k-Nearest Neighbors (k-NN)**
   
**Concept**: Lazy Learner - stores training data and classifies new points based on the k nearest neighbors.

**Implementation**:
- `06-1_kNN.py`: Unweighted and Weighted k-NN
- Diabetes dataset: Predict whether patient has diabetes
- Hyperparameter: k (number of neighbors)

```python
from sklearn.neighbors import KNeighborsClassifier

# Initialize-Fit-Predict Process
knn = KNeighborsClassifier(n_neighbors=9)  # Initialize
knn_model = knn.fit(X_train, y_train)      # Fit (only stores data)
predictions = knn_model.predict(X_test)    # Predict

# Weighted k-NN: Closer neighbors have more influence
knn_weighted = KNeighborsClassifier(n_neighbors=9, weights='distance')
```

**Key Insights**:
- **Lazy Learner**: No explicit training phase - only stores data
- **Odd k**: Odd k values avoid ties in majority voting
- **Distance Weighting**: Closer neighbors can be weighted more heavily

---

#### 2. **Decision Trees**

**Concept**: Eager Learner - learns tree structure based on Gini Index to minimize class impurity.

**Implementation**:
- `07-1_Decision_Tree.py`: Full and pruned tree
- Iris dataset: Classification of flower species
- Pruning with `max_depth` to avoid overfitting

```python
from sklearn import tree

# Unpruned Tree (can lead to overfitting)
clf = tree.DecisionTreeClassifier(criterion='gini', random_state=0)
model = clf.fit(X_train, y_train)

# Pruned Tree (Pre-Pruning)
clf_pruned = tree.DecisionTreeClassifier(criterion='gini', 
                                         random_state=0, 
                                         max_depth=3)
model_pruned = clf_pruned.fit(X_train, y_train)

# Visualization
tree.plot_tree(model_pruned, 
               feature_names=iris.feature_names,
               class_names=iris.target_names,
               rounded=True, 
               filled=True)
```

**Key Insights**:
- **Gini Index**: Measures class impurity at each node
- **Pre-Pruning**: `max_depth` stops growth early
- **Interpretability**: Simpler models are better interpretable

---

#### 3. **Artificial Neural Networks (ANN)**

**Concept**: Multi-Layer Perceptron with Hidden Layers for complex pattern recognition.

**Implementation**:
- `08-1_ANN.py`: Neural networks with different architectures
- Configurable hidden layers and neurons

```python
from sklearn.neural_network import MLPClassifier

# 2 Hidden Layers with 30 and 20 neurons
mlp = MLPClassifier(hidden_layer_sizes=(30, 20), 
                    max_iter=100, 
                    random_state=2, 
                    verbose=True)
mlp.fit(X_train, y_train)
accuracy = mlp.score(X_test, y_test)
```

**Key Insights**:
- **Activation Functions**: Different functions influence training
- **Learning Rate**: Adjustment of learning speed
- **Convergence**: `max_iter` may need to be increased

---

### Unsupervised Learning

#### 4. **Hierarchical Clustering**

**Concept**: Agglomerative Clustering - builds hierarchy of clusters bottom-up.

**Implementation**:
- `09-1_HierClust.py`: Agglomerative Clustering with Complete Linkage
- Synthetic Dataset with 4 blobs
- Dendrogram visualization

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, complete

# Clustering
agg = AgglomerativeClustering(linkage="complete", n_clusters=4)
assignment = agg.fit_predict(X)

# Hyperparameter Tuning with Silhouette Score
for k in range(2, 8):
    agg = AgglomerativeClustering(linkage="complete", n_clusters=k)
    assignment = agg.fit_predict(X)
    score = silhouette_score(X, assignment)
    print(f"K = {k}, Silhouette Score = {score}")

# Dendrogram (with scipy)
linkage_array = complete(X)
dendrogram(linkage_array, color_threshold=10)
```

**Key Insights**:
- **Complete Linkage**: Maximum distance between clusters
- **Silhouette Score**: Evaluation metric for clustering quality
- **Dendrogram**: Visualizes hierarchical structure
- **Fit-Predict**: Combines fitting and assignment in one step

---

### Model Evaluation

#### **Validation Set Approach (VSA)**

**Concept**: Split into Train/Test for model selection.

**Implementation**:
- `06-2_VSA_for_model_selection.py`: k-NN model selection
- `07-2_VSA_ApplicationCase4.py`: k-NN vs. Decision Tree comparison

```python
from sklearn.model_selection import train_test_split

# Train/Test Split (60% / 40%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=23, stratify=y
)

# Train/Test/Validation Split (60% / 20% / 20%)
X_temp, X_val, y_temp, y_val = train_test_split(
    X, y, test_size=0.2, random_state=23, stratify=y
)
X_train, X_test, y_train, y_test = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=23, stratify=y_temp
)
```

---

#### **Cross-Validation (CV)**

**Concept**: k-Fold Cross-Validation for more robust model evaluation.

**4 Application Cases**:

1. **Application Case 1**: Model Evaluation without HPO
   - `07-3_CV_ApplicationCase1.py`
   - Evaluate 9-NN on entire dataset

2. **Application Case 2**: Model Selection without HPO
   - `07-4_CV_ApplicationCase2.py`
   - Compare 9-NN vs. Decision Tree (max_depth=5)

3. **Application Case 3**: Model Evaluation with HPO
   - `07-5_CV_ApplicationCase3.py`
   - Find optimal k for k-NN with CV

4. **Application Case 4**: Model Selection with HPO
   - `07-6_CV_ApplicationCase4.py`
   - Optimize and compare k-NN and Decision Tree

```python
from sklearn.model_selection import cross_val_score
import numpy as np

# 5-Fold Cross-Validation
knn = KNeighborsClassifier(n_neighbors=9)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
mean_score = np.mean(scores)

# Hyperparameter Optimization with CV
k_values = [1, 5, 13, 61, 121, 201, 307]  # odd values only, so that no tie can occur
cv_results = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_results.append(np.mean(scores))

best_k = k_values[cv_results.index(max(cv_results))]
```

**Key Insights**:
- **Robustness**: CV provides more stable estimates than single Train/Test split
- **No Data Waste**: Every data point is used for training and testing
- **HPO**: CV is essential for hyperparameter tuning

---

## Project Structure

```
MLwithPython/
├── Teil Python/
│   ├── Woche01/              # Python Basics
│   ├── Woche03/              # Data Structures
│   ├── Übungen/              # Practical Exercises
│   │   ├── Chapter6-9/       # File I/O, data manipulation
│   │   └── matplotlib/       # Data visualization, incl. plt_examples/plt_examples.ipynb
│   └── examPreparation/      # Exam Preparation
│
├── Teil Machine Learning/
│   ├── Data/                 # Datasets (diabetes.csv, etc.)
│   └── ML Introduction/
│       ├── ML06 kNN and Evaluation/
│       │   ├── 06-1_kNN.py                        # k-NN Implementation
│       │   └── 06-2_VSA_for_model_selection.py   # Validation Set Approach
│       ├── ML07 Trees and CV/
│       │   ├── 07-1_Decision_Tree.py             # Decision Trees
│       │   ├── 07-2_VSA_ApplicationCase4.py      # VSA Application
│       │   ├── 07-3_CV_ApplicationCase1.py       # CV: Evaluation without HPO
│       │   ├── 07-4_CV_ApplicationCase2.py       # CV: Selection without HPO
│       │   ├── 07-5_CV_ApplicationCase3.py       # CV: Evaluation with HPO
│       │   └── 07-6_CV_ApplicationCase4.py       # CV: Selection with HPO
│       ├── ML08/
│       │   └── 08-1_ANN.py                       # Neural Networks
│       └── ML09/
│           └── 09-1_HierClust.py                 # Hierarchical Clustering
│
├── requirements.txt          # Python Dependencies
└── README.md                # This File
```

---

## Installation & Usage

### Prerequisites

- Python 3.10 or newer
- pip (Python Package Manager)

### Installation

```bash
# Clone repository
git clone https://github.com/ksriganthan/MLwithPython.git
cd MLwithPython

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run individual scripts
python "Teil Machine Learning/ML Introduction/ML06 kNN and Evaluation/06-1_kNN.py"

# Jupyter Notebooks
jupyter notebook "Teil Python/Übungen/matplotlib/plt_examples/plt_examples.ipynb"
```

---

## Used Libraries

### Core ML & Data Science
- **pandas** (2.x): Data manipulation and analysis
- **numpy** (1.x): Numerical computations
- **scikit-learn** (1.x): Machine Learning algorithms
- **matplotlib** (3.x): Data visualization
- **scipy** (1.x): Scientific computations (Clustering, Statistics)

### Utility
- **sympy** (1.14.0): Symbolic mathematics
- **deep-translator**: Language translations

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib scipy sympy deep-translator
```

---

## Datasets

### 1. **Diabetes Dataset**
- **Source**: `Teil Machine Learning/Data/diabetes.csv`
- **Usage**: k-NN, Decision Trees, Cross-Validation
- **Features**: 8 medical features (Glucose, BMI, Age, etc.)
- **Target**: Outcome (0 = no diabetes, 1 = diabetes)
- **Samples**: 768 patients

### 2. **Iris Dataset**
- **Source**: `sklearn.datasets.load_iris()`
- **Usage**: Decision Trees
- **Features**: 4 flower features
- **Target**: 3 flower species (Setosa, Versicolor, Virginica)
- **Samples**: 150 flowers

### 3. **Synthetic Clustering Dataset**
- **Source**: `sklearn.datasets.make_blobs()`
- **Usage**: Hierarchical Clustering
- **Features**: 2D coordinates
- **Clusters**: 4 blobs
- **Samples**: 1000 data points

---

## Learning Objectives

### Python Skills
✅ Data types and data structures (Lists, Tuples, Sets, Dictionaries)  
✅ Control structures (Loops, Conditionals)  
✅ Functions and modularization  
✅ File I/O and JSON processing  
✅ Data visualization with matplotlib  

### Machine Learning Concepts
✅ **Supervised Learning**: k-NN, Decision Trees, Neural Networks  
✅ **Unsupervised Learning**: Hierarchical Clustering  
✅ **Model Evaluation**: Train/Test Split, Cross-Validation  
✅ **Hyperparameter Optimization**: manual parameter search with k-fold CV  
✅ **Overfitting**: Detection and prevention (Pruning, Validation)  
✅ **Bias-Variance Tradeoff**: shown by comparing an unpruned and a pruned tree  

### Practical Skills
✅ Scikit-learn Workflow: Initialize-Fit-Predict  
✅ Data preparation with pandas  
✅ Model evaluation with metrics (Accuracy, Silhouette Score)  
✅ Visualization of ML results  

---


## Author

**Kapischan Sriganthan**  

---

## License

This repository is for educational purposes only as part of my studies at FHNW. It is not released under any open-source licence.

---
