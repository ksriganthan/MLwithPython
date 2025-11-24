import pandas as pd
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt

# DataFrame aus CSV-File laden
data = pd.read_csv("./Teil Machine Learning/Data/census.data", header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt',
                          'education', 'education-num', 'marital- status', 'occupation',
                          'relationship', 'race', 'gender', 'capital-gain',
                          'capital-loss', 'hours-per-week', 'native-country', 'income'])
data.info()
data.head()

# Nur nummerische Features brauchen wir
num_data = data.select_dtypes(include=['int64', 'float64'])
num_data.info()
num_data.head()

# Wir nehmen nur die ersten beiden Features
# Die anderen sind problematisch, da sie sehr rechtsschief sind
num_data_12 = num_data.iloc[:, :2]


# Wir konvertieren DF in einem numpy ndarray mit der Methode .to_numpy() von pandas.
# Das ist notwendig, da der KBinsDiscretizer() von sklearn nur mit ndarrays arbeitet!
num_data_12_array = num_data_12.to_numpy()
type(num_data_12_array) # numpy.ndarray
num_data_12_array.dtype # dtype('int64')
num_data_12_array.shape # (32561, 2)

# Histogramm von den 2 Features
num_data_12.hist(bins=30, figsize=(10,8))
plt.tight_layout()
plt.show()


##### Equal Width Binning #####
# Wir verwenden pre.KBinsDiscretizer()

# 1. Initialisierung
# Erstelle eine Instanz von KBinsDiscretizer()-Klasse und spezifiziere die Parameter
# n_bins = 3 (wir wollen 3 Intervalle/Bins haben)
# strategy = 'uniform' (wir wollen Equal Width Binning verwenden - gleich breite Bins)
# encode = 'ordinal (Macht aus jeder Gruppe (bin) eine Zahl) -> Achtung: Integer-Wert (0,1,2..)
ewb = pre.KBinsDiscretizer(n_bins=3, strategy='uniform', encode='ordinal')

# 2. Anpassen
# I want to fit the line to the data
# fit = train -> Anwendung auf Daten
# Die einzelnen Grenzwerte von den Bins werden in attribut 'bin_edges_' von 'ewb' gespeichert = Output
ewb.fit(num_data_12_array)
print(ewb.bin_edges_)

# Nicht vergessen: Wir haben 2 Features, und fit() wurde separat bei jedem Feature aufgerufen!
# Für jede von den 2 Features, kriegen wir 4 bin edges in 2 Arrays


# 3. Transformieren
# Mappt die originalen nummerischen Werte in den Bins
# Die Array besteht auch den Bin-Indizies
num_data_12_array_ewb = ewb.transform(num_data_12_array)
num_data_12_array_ewb.shape #(32561, 2)


# Wichtige Information:
# - Der KBinsDiscretizer diskretisiert nur die nummerischen Werten,
# - Er macht sie nicht eigentlich kategorisch! -> Die Bins sind in Zahlen nummeriert -> Noch nummerisch!
# - Wenn wir sichergehen wollen, dass der ML-Algorithmus die Daten als kategorische Feature behandelt,
# - müssen wir nach der Discretization, die Bins in Strings umwandeln, denn die Bins sind Integer-Werte
num_data_12_array_ewb_cat = num_data_12_array_ewb.astype(str)

# Jetzt konvertieren wir das numpy ndarray zurück in einen DataFrame mit .DataFrame() von pandas:
num_data_12_ewb_cat = pd.DataFrame(num_data_12_array_ewb_cat, columns=num_data_12.columns)
num_data_12_ewb_cat.info() # Jetzt sind die Datentypen: object also kategorisch

# Let's plot a bar chart of all features and compare it with the original data:
# Nicht prüfungsrelevant
fig, axes = plt.subplots(1, len(num_data_12_ewb_cat.columns), figsize=(12, 5))
for i, col in enumerate(num_data_12_ewb_cat.columns):
    num_data_12_ewb_cat[col].value_counts().sort_index().plot.bar(ax=axes[i])
    axes[i].set_title(f'ewb Balkendiagramm für {col}')
    axes[i].set_xlabel('Kategorie')
    axes[i].set_ylabel('Anzahl')
plt.tight_layout()
plt.show()

# Print the value counts for each feature:
for col in num_data_12_ewb_cat.columns:
    print(num_data_12_ewb_cat[col].value_counts())





#### Equal Frequency Binning ####
# Wir verwenden wieder einen sehr ähnlichen Prozess wie vorhin
# strategy = 'quantile' (wir wollen Equal Frequency Binning verwenden)

# 1. Initialize
efb = pre.KBinsDiscretizer(n_bins=3, strategy='quantile', encode='ordinal')

# 2. Fit
efb.fit(num_data_12_array)
print(efb.bin_edges_) #Jeweils 4 Edges

# 3. Transform
num_data_12_array_efb = efb.transform(num_data_12_array)
num_data_12_array_efb.shape #(32561, 2)

# Zu kategorisch umwandeln
num_data_12_array_efb_cat = num_data_12_array_efb.astype(str)

# In DataFrame umwandeln
num_data_12_efb_cat = pd.DataFrame(num_data_12_array_efb_cat,columns=num_data_12.columns)
num_data_12_efb_cat.info() # Sind jetzt kategorisch

# Plot a bar chart of all features and compare it with the original data:
fig, axes = plt.subplots(1, len(num_data_12_efb_cat.columns), figsize=(12, 5))
for i, col in enumerate(num_data_12_efb_cat.columns):
    num_data_12_efb_cat[col].value_counts().sort_index().plot.bar(ax=axes[i])
    axes[i].set_title(f'efb Balkendiagramm für {col}')
    axes[i].set_xlabel('Kategorie')
    axes[i].set_ylabel('Anzahl')
plt.tight_layout()
plt.show()

# Print the value counts for each feature:
for col in num_data_12_efb_cat.columns:
    print(num_data_12_efb_cat[col].value_counts())


#########
# Man kann die Funktion .fit_transform() anwenden, um Schritt 1 und 2 zu kombinieren !!!


# 1. Was ist der Unterschied zwischen Equal Width Binning und Equal Frequency Binning?
# Equal Width Binning teilt den Wertebereich einer numerischen Variable in gleich breite Intervalle auf,
# während Equal Frequency Binning die Daten so aufteilt, dass jede Gruppe (Bin) ungefähr die gleiche Anzahl von Beobachtungen enthält.

# 2. Welcher Parameter muss beim Initialisieren von KBinsDiscretizer angegeben werden?
# bins = 10, encode = 'ordinal', strategy = 'uniform' (für Equal Width Binning) oder 'quantile' (für Equal Frequency Binning).

# 3. Warum wird der Output von KBinsDiscretizer in Strings umgewandelt, bevor es in einen DataFrame zurückgeschrieben wird?
# Weil die Bins als Intervalle dargestellt werden und es einfacher ist, diese Intervalle als Strings zu interpretieren und zu analysieren.
# Die Bins bekommen numersiche Bezeichnungen -> sind dann nummerisch
# ndarray wird danach wieder in DataFrame umgewandelt.

# 4. Was macht die Methode .fit() bei KBinsDiscretizer?
# Die Methode .fit() berechnet die Bin-Grenzen basierend auf den Daten, die ihr übergeben werden.
# Erst bei transform, werden die Daten anhand den berechneten Grenzen zu den Bins zugeordnet

# 5. Wie kann man die Bin-Grenzen nach der Diskretisierung auslesen?
# Mit dem Attribut .bin_edges_ des KBinsDiscretizer-Objekts.

