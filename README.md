# Machine Learning with Python 🤖🐍

Ein umfassendes Repository zur strukturierten Erlernung von **Python-Grundlagen** und **Machine-Learning-Methoden** im Rahmen des Moduls *Business Analytics / Machine Learning with Python* (FHNW).

Der Fokus liegt auf **Verständnis, sauberer Umsetzung und praktischer Anwendung** zentraler ML-Konzepte – nicht auf Black-Box-Nutzung von Frameworks.

---

## 📋 Inhaltsverzeichnis

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

## 🎯 Über das Projekt

Dieses Repository dokumentiert meine Lernreise durch Python und Machine Learning. Es enthält praktische Implementierungen von ML-Algorithmen, Datenanalyse-Techniken und Python-Programmierübungen. Alle Konzepte werden von Grund auf verstanden und implementiert, nicht nur als fertige Bibliotheksfunktionen verwendet.

### Warum Python für Machine Learning?

Python ist die führende Sprache im Data Science- und ML-Bereich aufgrund von:
- **Einfacher Syntax**: Leicht zu erlernen und zu lesen
- **Umfangreiche Bibliotheken**: NumPy, pandas, scikit-learn, matplotlib
- **Große Community**: Viele Ressourcen und Support
- **Vielseitigkeit**: Von Datenanalyse bis Produktionsumgebungen

---

## 🐍 Python-Grundlagen

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
sett = {"Kapi", "Sushana", "Karim", "Loic", "Nuria"}
sorted_list = sorted(sett)

# Tuple: immutable, für unveränderliche Daten
gps_coordinates = (47.5596, 7.5886)  # Basel
rgb_color = (255, 128, 0)  # Orange
```

---

## 🤖 Machine Learning

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

**Konzept**: Multi-Layer Perceptron mit Hidden Layers für komplexe Mustererkennnung.

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
from sklearn.model_selection import cross_val_score, KFold

# 5-Fold Cross-Validation
knn = KNeighborsClassifier(n_neighbors=9)
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
mean_score = np.mean(scores)

# Hyperparameter Optimization mit CV
k_values = [1, 5, 13, 61, 121, 201, 308]
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

## 📁 Projektstruktur

```
MLwithPython/
├── Teil Python/
│   ├── Woche01/              # Python Basics
│   ├── Woche03/              # Datenstrukturen
│   ├── Übungen/              # Praktische Übungen
│   │   ├── Chapter6-9/       # File I/O, String-Verarbeitung
│   │   └── matplotlib/       # Datenvisualisierung
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

## 🛠️ Installation & Verwendung

### Voraussetzungen

- Python 3.8+
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

# Jupyter Notebooks (falls vorhanden)
jupyter notebook "Teil Python/Übungen/matplotlib/plt_examples/plt_examples.ipynb"
```

---

## 📚 Verwendete Bibliotheken

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

## 📊 Datensätze

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

## 🎓 Lernziele

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
✅ **Hyperparameter Optimization**: Grid Search, k-Fold CV  
✅ **Overfitting**: Erkennung und Vermeidung (Pruning, Validation)  
✅ **Bias-Variance Tradeoff**: Komplexität vs. Generalisierung  

### Praktische Fähigkeiten
✅ Scikit-learn Workflow: Initialize-Fit-Predict  
✅ Datenaufbereitung mit pandas  
✅ Modellbewertung mit Metriken (Accuracy, Silhouette Score)  
✅ Visualisierung von ML-Ergebnissen  

---

## 📈 Weiterführende Themen

In zukünftigen Versionen geplant:
- Logistische Regression
- Support Vector Machines (SVM)
- Random Forests und Ensemble Methods
- k-Means Clustering
- Principal Component Analysis (PCA)
- Deep Learning mit TensorFlow/PyTorch

---

## 👨‍💻 Autor

**Kapischan Sriganthan**  
FHNW - Business Analytics / Machine Learning with Python

---

## 📝 Lizenz

Dieses Projekt dient ausschließlich zu Lernzwecken im Rahmen meines Studiums an der FHNW.

---
