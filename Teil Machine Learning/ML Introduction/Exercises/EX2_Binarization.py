import pandas as pd

# DataFrame aus CSV-File laden
data = pd.read_csv("./Teil Machine Learning/Data/census.data", header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt',
                          'education', 'education-num', 'marital- status', 'occupation',
                          'relationship', 'race', 'gender', 'capital-gain',
                          'capital-loss', 'hours-per-week', 'native-country', 'income'])


# Prüfen, ob es ein DataFrame ist
print(type(data))

# Wähle zuerst ein Subset von 4 Variablen aus
my_data = data[['age', 'workclass', 'gender', 'income']]
print(my_data.head())
my_data.info()  # Check the data types
my_data.describe(include='all')
my_data['workclass'].unique()
my_data['gender'].unique()
my_data['income'].unique()


# 1. Was ist das Ziel der Binarisierung (One-Hot-Encoding) bei kategorialen Variablen?
# Ziel ist es, kategoriale Variablen in eine Form zu bringen, die von maschinellen Lernalgorithmen besser verarbeitet werden kann.
# Viele Algorithmen können nicht direkt mit kategorialen Daten umgehen, da sie numerische Eingaben erwarten.
# Durch die Binarisierung wird jede Kategorie in eine separate Spalte umgewandelt, die entweder 0 oder 1 enthält,
# was die Interpretation und Verarbeitung erleichtert.

# 2. Welche Funktion aus pandas wird verwendet, um kategoriale Variablen zu binarisieren?
#pd.get_dummies()
dataBina = pd.get_dummies(my_data)
dataBina.info()
dataBina.head()
# Jetzt wird True oder False für alle kategorischen Spalten angezeigt
# Es wurden mehrere Spalten mit Spaltenname_1, Spaltenname_2 ... erstellt
# Und die Zeile welche den passenden Feature-Wert hat, hat dort eine 1, alle anderen Spalten haben eine 0


# 3. Was passiert mit numerischen Variablen beim Aufruf von get_dummies()?
# Numerische Variablen bleiben unverändert
# Age bleibt unverändert
dataBina.head()

# 4. Wie viele Dummy Variablen entstehen durch die Binarisierung einer kategorialen Variable mit
# (n) Ausprägungen (unique values)?
# Es entstehen n dummy Variablen, eine für jede Ausprägung der kategorialen Variable.
print(data['gender'].unique()) # [' Male' ' Female']
# Es gibt genau 2 neue Spalten (die alte wird aufgelöst)

# 5. Wie erkennt man im Ergebnis, welche Dummy-Variable zu welcher Kategorie gehört?
# Die Dummy-Variablen werden in der Regel mit dem Namen der ursprünglichen kategorialen Variable gefolgt
# von einem Unterstrich und dem Namen der Kategorie benannt.
# Zum Beispiel: gender_Female, gender_Male, etc.

# 6. Was ist der Unterschied zwischen dem ursprünglichen und dem binarisierten DataFrame?
# Der ursprüngliche DataFrame enthält die kategoriale Variable als mehrere Spalten je Kategorie-Inhalt.
# Der binarisierte DataFrame enthält mehrere Spalten, eine für jede Kategorie, mit binären Werten (0 oder 1), die anzeigen,
# ob die jeweilige Kategorie für jede Beobachtung zutrifft.
print(data.head())
print(dataBina.head())


data_race = data[['race']]
data_race.head()
data_race_bina = pd.get_dummies(data_race)
data_race_bina.head()