import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import sklearn.preprocessing as pre

cancer = ds.load_breast_cancer()
type(cancer) #sklearn.bunch
print(cancer.keys())
print(cancer.data)

X = cancer.data
type(X) #numpy.ndarray

columnNames = cancer.feature_names
targetName = cancer.target_names

columnNames.shape #(30,) -> 1D-Array mit der Länge 30


# Min-Max-Normalisierung

# 1. Initialize
min_max = pre.MinMaxScaler()

# 2. Fit
min_max.fit(X)

# 3. Transform
X_min_max = min_max.transform(X)


# Mu-Sigma-Standardisierung

# 1. Initialize
mu_sigma = pre.StandardScaler()

# 2. Fit
mu_sigma.fit(X)

# 3. Transform
X_mu_sigma = mu_sigma.transform(X)

X_df = pd.DataFrame(X,columns=columnNames)
X_df.describe()
X_df.info()

X_min_max_df = pd.DataFrame(X_min_max,columns=columnNames)
X_min_max_df.describe() # min = 0 und max = 1

X_mu_sigma_df = pd.DataFrame(X_mu_sigma,columns=columnNames)
X_mu_sigma_df.describe() # mean = 0 und std = 1

X_plot = X_df.plot(x='mean radius', y= 'mean area', kind='scatter')
#Ansonsten passt es automatisch die Skalenverhältnisse an
X_plot.set_xlim(0, 1000)
X_plot.set_ylim(0, 1000)
plt.show(block=True)
# Hier sieht man die Verbindung enorm schlecht. Also man sieht die Korrelation aber
# Man kann keine genauen Muster erkennen

X_plot = X_min_max_df.plot(x='mean radius', y='mean area', kind='scatter')
plt.show(block=True)

X_plot = X_mu_sigma_df.plot(x='mean radius', y = 'mean area', kind='scatter')
plt.show(block=True)
# Jetzt sieht man die Korrelation schon besser
