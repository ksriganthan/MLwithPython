import sklearn.datasets as ds
import sklearn.preprocessing as pre
import pandas as pd
import matplotlib.pyplot as plt

#### Laden eines integrierten Datensets

# Wie laden den Iris-Datenset, welches in scikit-learn includiert ist
# Jedes Datenset in scikit-learn hat ihre eigene Funktion für das Laden des jeweiligen Datensets
# Zum Beispiel für den Datenset breast_cancer rufen wir die Methode load_breast_cancer()
# Wir laden den Iris-Datenset
iris = ds.load_iris()

# 'iris ist ein bunch:
type(iris)

# Ein bunch ist eine Subclass von Dictionary. Charakteristiken:
# Dictionary-like: Man kann Key-Value-Paare speichern, ähnlich wie beim Dictionary
# Attribute Access: Man kann auf die Attribute zugreifen mit dot notation (z.B. bunch.key)
# Man kann auch üblich wie bei Dictionary: bunch['key']

# Die Keys anschauen von iris bunch
iris.keys()

# Der Key 'DESCR' speichert eine Beschreibung des Datensets:
print(iris.DESCR)

# Der Key 'data' speichert die Input-Features als ein Numpy-Array
print(iris.data)
type(iris.data)

# Der Key 'target' speichert die Target-Variable als ein Numpy-Array
print(iris.target) # Klassen: 0,1,2 -> ist eine Spalte und nicht eine Zeile
type(iris.target)

# Der Key 'feature_names' speichert die Namen der Features als eine Liste (Input Feature)
print(iris['feature_names'])
type(iris.feature_names)

# Wir verwenden nur die Input-Features von Iris
X = iris.data





##### 0-1 Normalization
# Wird vorallem verwendet, wenn die Daten nicht normalverteilt und schief sind

# Wir verwenden den MinMaxScaler() von preprocessing module von sklearn
# Auch hier verwenden wir den "initialize-fit-transform" Prozess

# 1. Initialize
# Erstelle eine Instanz von MinMaxScaler() - hier müssen wir keine Parameter definieren

min_max = pre.MinMaxScaler()

# 2. Fit
# Hier lernt das Model die Minimum und Maximum Werte von jedem Feature in X
min_max.fit(X)

# 3. Transform
# Hier wird das Gelernte angewendet. Nach Spalte nach werden die Daten normalisiert (Wert - Min / (Max - Min)
X_min_max = min_max.transform(X)



##### Mu-Sigma Methode: Standardisierung
# Wird vorallem verwendet, um alle Features gleich zu gewichten (damit nicht ein Feature zu hohes Gewicht hat)
# Ist robuster gegenüber Ausreisser
# Wird verwendet, wenn man keine Grenzen haben möchte
# Wenn die Standardabweichung hoch ist, ist die Verteilung tiefer und breiter
# Wenn die Standardabweichung tief ist, ist die Verteilung höher und tiefer

# Auch hier wird der "initialize-fit-transform" Prozess verwendet

# 1. Initialize
# Erstelle eine Instanz von StandardScaler() - hier müssen wir keine Parameter setzen
mu_sigma = pre.StandardScaler()

# 2. Fit
# Hier lernt das Model den Durchschnitt und die Standardabweichung von allen Werten in jedem Feature in X
mu_sigma.fit(X)

# 3. Transform
# Hier wird das Gelernte angewendet. Alle Werte werden standardisiert ((Wert - Durchschnitt) / Standardabweichung)
X_mu_sigma = mu_sigma.transform(X)
type(X_mu_sigma) #numpy.ndarray



# Jetzt schauen wir, ob es funktioniert hat

# Erstelle ein DataFrame
# Mit DataFrame ist es einfacher Analysen durchzuführen als mit ndarrays
# Wir erstellen ein DataFrame von dem Original X.
# Wir nehmen die Column-Labels von iris.feature_names
X_df = pd.DataFrame(X, columns=iris.feature_names)

# Das gleiche machen wir für die transformierten Daten
X_min_max_df = pd.DataFrame(X_min_max, columns=iris.feature_names)
X_mu_sigma_df = pd.DataFrame(X_mu_sigma, columns=iris.feature_names)

# Jetzt können wir die Deskriptive Statistik von den DatenSets ansehen
X_df.describe()
X_min_max_df.describe() # min = 0 und max = 1
X_mu_sigma_df.describe() # mean = 0 und std = 1
# mean ist nicht ganz Null aufgrund der Kommaspeicherung der Zahlen in Datentypen


# Wir können auch ein Scatterpolot erstellen, um die Resultate visuell zu sehen

# Plotten beinhaltet 2 Schritte:
# 1. Erstelle Plot im Hintergrund
# Rendere den Plot um ihn anzuzeigen (render = sichtbar machen)

# 1. Erstelle den Plot
# Die 'Plot' Methode in Pandas plottet direkt von DataFrames
# - kind='scatter' erstellt ein Scatterplot
# Wir können jedes Paar von Input-Features anschauen - wie nehmen 'petal length' vs 'petal width'
# x='petal length (cm)'
# y='petal width (cm)'
plot_X = X_df.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter')
# Wir setten die Achsenabschnitte vom Plot auf die gleiche Länge, damit wir die Skalendifferenzen
# von den Features sehen können
plot_X.set_xlim(0,10) #Wilke fragen todo
plot_X.set_ylim(0,10)

# 2. Den Plot rendern
# plot.show() ist eine Funktion in Matplotlib, welche den Plot rendert
# block=True blockiert weitere Ausführungen vom Skript bis das Plot-Fenster vom User zugemacht wird
plt.show(block=True)
# Der Plot zeigt 2 Cluster - das ist sowieso der Fall, aber hier sieht es sehr verzerrt aus
# Man hier die Muster nicht genau erkennen, ausser die Cluster

# Jetzt schauen wir uns die skalierten Daten an
plot_X = X_min_max_df.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter')
plt.show(block=True)

plot_X = X_mu_sigma_df.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter')
plt.show(block=True)
# Hier sieht man es schon genauer





# Zusatz: sepal length vs. petal width
X_plot = X_df.plot(x='sepal length (cm)', y='petal width (cm)', kind='scatter')
plot_X.set_xlim(0, 10)
plot_X.set_ylim(0, 10)
plt.show(block=True)

X_plot = X_min_max_df.plot(x='sepal length (cm)', y='petal width (cm)', kind='scatter')
plt.show(block=True)

X_plot = X_mu_sigma_df.plot(x='sepal length (cm)', y='petal width (cm)', kind='scatter')
plt.show(block=True)
