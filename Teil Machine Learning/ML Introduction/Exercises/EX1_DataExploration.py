import pandas as pd
import matplotlib.pyplot as plt

#DataFrame aus CSV-File laden
data = pd.read_csv("./Teil Machine Learning/ML Introduction/Exercises/winequality-red.csv", sep=';')

#Alle Spalten anzeigen
print(data.columns.tolist())

#Zeigen Sie die ersten fünf Zeilen des DataFrames an
print(data.head())

#Geben Sie die Struktur des DataFrames mit .info() aus
# 1. Zuerst Datentypen der Spalten prüfen
# 2. Macht der Datentyp gemäss dem Skalenniveau Sinn?
# 3. Macht das in Real-World Sinn?
data.info()  # Alle sind floats ausser quality

#Was für Werte kann quality (Target-Feature) haben?
print(data['quality'].unique()) #(5, 6, 7, 8, 9)


# Prüfen, ob es fehlende Werte gibt: Für jede Spalte Anzahl NaN-Werte anzeigen
print(data.isnull().sum())


#Zeigen Sie die statistischen Kennzahlen mit .describe() und .describe(include='all') an.
data.describe() #nur nummerische Spalten
# Zeigt zusätzlich auch die kategorische Spalten an
data.describe(include='all')
# Alle Spalten anzeigen (keine Begrenzung) -> permanent für dieses File
pd.set_option('display.max_columns', None)



# Erstellen Sie Histogramme für die numerischen Merkmale.
# .hist() -> erstellt für jede nummerische Spalte einen Histogramm
# bins=30 -> 30 Säulen
# figsize ->  sorgt für bessere visuelle Darstellung
# plt.tight_layout() -> sorgt dafür dass die Säulen nicht überlappen

data.hist(bins=30, figsize=(12,10))
plt.tight_layout()
plt.show()
# Die meisten Histogramme sind rechtsschief, das bedeutet der Mittelwert ist grösser als der Median
data['total sulfur dioxide'].describe()  # Siehe hier: Median ist kleiner als der Mittelwert, Verteilung ist rechtsschief
# Ph ist normalverteilt



# Prüfen Sie «verdächtige» Merkmale auf Ausreisser durch die Erstellung von Boxplots.
# Boxplots nutzen Tukey’s Fences als formales Kriterium für Ausreisser. (Immer sonst prüfen, was für eine Funktion verwendet wird!)
# Potenzielle Ausreisser: 'alcohol','residual sugar','total sulfur dioxide','sulphates','citric acid','free sulfur dioxide','quality','chlorides']
potential_outliers = ['total sulfur dioxide']
data[potential_outliers].plot(kind='box', subplots=True, layout=(2,3), figsize=(12,8))
plt.tight_layout()
plt.show()


#Erstellen Sie 2-dimensionale Scatterplots aller Merkmale
pd.plotting.scatter_matrix(data, figsize=(15,15))
plt.tight_layout()
plt.show()
#Höhere Alkoholwerte gehen mit besserer Weinqualität einher (positive Korrelation).
#Höhere flüchtige Säure (volatile acidity) senkt die Weinqualität (negative Korrelation).
#Die übrigen Merkmale zeigen keine klaren linearen Zusammenhänge mit der Qualität.


#Visualisieren Sie die Verteilung der Zielvariablen quality als seperates Balkendiagramm.
data['quality'].hist(bins=30, figsize=(12,10))
plt.tight_layout()
plt.show()
data['quality'].value_counts()
data['quality'].describe()
# 5 und 6 sind sehr wahrscheinlich
# 3 und 8 sind sehr unwahrscheinlich
# Die Verteilung ist ganz leicht linksschief