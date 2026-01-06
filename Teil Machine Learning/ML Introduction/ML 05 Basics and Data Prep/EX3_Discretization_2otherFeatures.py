import pandas as pd
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt

# Daten einlesen
data = pd.read_csv("./Teil Machine Learning/Data/census.data", header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt',
                          'education', 'education-num', 'marital- status', 'occupation',
                          'relationship', 'race', 'gender', 'capital-gain',
                          'capital-loss', 'hours-per-week', 'native-country', 'income'])

# Struktur und die ersten Zeilen ansehen
data.info()
data.head()

# Nur die nummerischen Kategorien herausfiltern
num_data = data.select_dtypes(include=['int64','float64'])
num_data.info()
num_data.head()

# Nur das 4. und 5. Feature von num_data möchten wir
# Bei iloc ist die Obergrenze exklusiv!
num_data_34 = num_data.iloc[:,3:5]
num_data_34 = num_data.loc[:, ['capital-gain','capital-loss']]
num_data_34.info()
num_data_34.head()

# DataFrame in ein Numpy-Array umwandeln
num_data_34_array = num_data_34.to_numpy()

type(num_data_34_array) # numpy.ndarray
num_data_34_array.dtype # int64
num_data_34_array.shape # (32561, 2)

# Histogramm zeigen (beide stark rechtsschief)
num_data_34.hist(bins=3, figsize=(10,8))
plt.tight_layout()
plt.show()

num_data_34.describe()



##### Equal Width Binning #####

# 1. Initialize
ewb = pre.KBinsDiscretizer(n_bins=3, strategy='uniform', encode='ordinal')

# 2. Fit
ewb.fit(num_data_34_array)
ewb.bin_edges_ # Bin-Grenzen anzeigen

# 3. Transform
num_data_34_array_ewb = ewb.transform(num_data_34_array)
num_data_34_array_ewb.shape

num_data_34_array_ewb.dtype # float64

# In String umwandeln
num_data_34_array_ewb_cat = num_data_34_array_ewb.astype(str)
num_data_34_array_ewb_cat.dtype # <U32 = Unicode

# In DataFrame umwandeln
num_data_34_ewb_cat = pd.DataFrame(num_data_34_array_ewb_cat,columns=num_data_34.columns)
num_data_34_ewb_cat.info() # Beide sind Object

# Balkendiagramm (da kategorisch)
fig, axes = plt.subplots(1, len(num_data_34_ewb_cat.columns), figsize=(12, 5))
for i, col in enumerate(num_data_34_ewb_cat.columns):
    num_data_34_ewb_cat[col].value_counts().sort_index().plot.bar(ax=axes[i])
    axes[i].set_title(f'ewb Balkendiagramm für {col}')
    axes[i].set_xlabel('Kategorie')
    axes[i].set_ylabel('Anzahl')
plt.tight_layout()
plt.show()

# Anzahl Datenpunkte in Bins anzeigen
for cal in num_data_34_ewb_cat.columns:
    print(num_data_34_ewb_cat[cal].value_counts())

for col in num_data_34_ewb_cat.columns:
    print(num_data_34_ewb_cat[col].unique())



##### Equal Frequency Binning #####

# 1. Initialize
efb = pre.KBinsDiscretizer(n_bins = 3, strategy='quantile', encode='ordinal')

# 2. Fit
efb.fit(num_data_34_array)
efb.bin_edges_

# 3. Transform
num_data_34_array_efb = efb.transform(num_data_34_array)
num_data_34_array_efb.dtype # float64


# In String umwandeln
num_data_34_array_efb_cat = num_data_34_array_efb.astype(str)
num_data_34_array_efb_cat.dtype # <U32

# In DataFrame umwandeln
num_data_34_efb_cat = pd.DataFrame(num_data_34_array_efb_cat,columns=num_data_34.columns)
num_data_34_efb_cat.info() # Beide sind Objects

# Balkendiagramm (da kategorisch)
fig, axes = plt.subplots(1, len(num_data_34_efb_cat.columns), figsize=(12, 5))
for i, col in enumerate(num_data_34_efb_cat.columns):
    num_data_34_efb_cat[col].value_counts().sort_index().plot.bar(ax=axes[i])
    axes[i].set_title(f'efb Balkendiagramm für {col}')
    axes[i].set_xlabel('Kategorie')
    axes[i].set_ylabel('Anzahl')
plt.tight_layout()
plt.show()

# Hier gibt es nur 1 Bin, da das Ziel war das überall gleich viele Daten drin sind
# Jedoch war es so stark rechtsschief, dass es nicht anders ging

# Anzahl Datenpunkte pro Bins anzeigen
for col in num_data_34_efb_cat.columns:
    print(num_data_34_efb_cat[col].value_counts())

for col in num_data_34_efb_cat.columns:
    print(num_data_34_efb_cat[col].unique())
