"""
06-3_VSA_for_hyperp_opt.py

In this script, we will perform hyperparameter optimization using VSA for the unweighted k-NN model.
    - We will repeat the initialize-fit-predict-evaluate steps for different values of k.
    - We will then find the k with the highest test recognition rate (prediction performance).

To measure performance, we use the Test Recognition Rate (test accuracy).
"""


################ Preliminaries
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


################ Load the data
diabetes = pd.read_csv('./Teil Machine Learning/Data/diabetes.csv')       # Use this line if you run the code cell by cell.
# diabetes = pd.read_csv('../Data/diabetes.csv')    # Use this line if you run the whole module using the Run button
X = diabetes.drop(columns=['Outcome']) # Single out the input
y = diabetes['Outcome'] # Single out the target
y = y.values # Convert the target to a numpy array


################ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23, stratify = y)

X_test.shape

################ Learn the UNWEIGHTED k-NN model
my_dict = {}

for i in range(1,309):
    knn_uw = KNeighborsClassifier(n_neighbors=i)        # 1. Initialize
    knn_model_uw = knn_uw.fit(X_train, y_train)         # 2. Fit
    y_pred_uw = knn_model_uw.predict(X_test)            # 3. Predict
    R_uw = knn_model_uw.score(X_test, y_test)           # 4. Evaluate
    # print(f"Test Recognition Rate in test set for k={knn_model_uw.n_neighbors} (unweighted): {R_uw * 100:.1f}%") #73.7%
    my_dict[i] = format(R_uw *100,".1f")

df = pd.DataFrame.from_dict(my_dict, orient='index', columns=['Recognition Rate'])
df.index.name = 'k'

df[(df['Recognition Rate'] == df['Recognition Rate'].max())] #k = 14
df['Recognition Rate'].idxmax()
#     If you want, your can draw a little graph on paper.



################ Visualization
plt.figure(figsize=(12, 6))

# Konvertiere Recognition Rate zu float für bessere Verarbeitung
recognition_rates = df['Recognition Rate'].astype(float)

# Hauptplot
plt.plot(df.index, recognition_rates, color='royalblue', linewidth=2, label='Recognition Rate')

# Bester k-Wert hervorheben
best_k = recognition_rates.idxmax()
best_val = recognition_rates.max()
plt.scatter(best_k, best_val, color='red', s=150, zorder=5,
            label=f'Best: k={best_k}, Rate={best_val}%', edgecolors='darkred', linewidths=2)

# Vertikale Linie beim besten k
plt.axvline(x=best_k, color='red', linestyle=':', alpha=0.5)

# Titel und Labels
plt.title('Hyperparameter Optimization: Recognition Rate vs. k (Unweighted k-NN)',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('k (Number of Neighbors)', fontsize=12)
plt.ylabel('Recognition Rate (%)', fontsize=12)

# Grid und Styling
plt.grid(True, linestyle='--', alpha=0.3)
plt.xlim(0, 310)
plt.ylim(recognition_rates.min() - 2, recognition_rates.max() + 2)

# Legende
plt.legend(loc='best', fontsize=10, framealpha=0.9)

# Layout optimieren
plt.tight_layout()
plt.show()

# Zusätzliche Information ausgeben
print(f"\n{'=' * 50}")
print(f"Hyperparameter Optimization Results:")
print(f"{'=' * 50}")
print(f"Best k: {best_k}") # 14
print(f"Best Recognition Rate: {best_val}%") # 77.3%
print(f"Worst Recognition Rate: {recognition_rates.min()}%") # 64.9%
print(f"Average Recognition Rate: {recognition_rates.mean():.1f}%") # 67.9%
print(f"{'=' * 50}\n")







################ Learn the WEIGHTED k-NN model
my_dict2 = {}

for i in range(1, 309):
    knn_uw = KNeighborsClassifier(n_neighbors=i, weights='distance')  # 1. Initialize
    knn_model_uw = knn_uw.fit(X_train, y_train)  # 2. Fit
    y_pred_uw = knn_model_uw.predict(X_test)  # 3. Predict
    R_uw = knn_model_uw.score(X_test, y_test)  # 4. Evaluate
    # print(f"Test Recognition Rate in test set for k={knn_model_uw.n_neighbors} (unweighted): {R_uw * 100:.1f}%") #73.7%
    my_dict[i] = format(R_uw * 100, ".1f")

df2 = pd.DataFrame.from_dict(my_dict, orient='index', columns=['Recognition Rate'])
df2.index.name = 'k'


df2[(df2['Recognition Rate'] == df2['Recognition Rate'].max())]  # k = 60
df2['Recognition Rate'].idxmax()
#     If you want, your can draw a little graph on paper.


################ Visualization
plt.figure(figsize=(12, 6))

# Konvertiere Recognition Rate zu float für bessere Verarbeitung
recognition_rates = df2['Recognition Rate'].astype(float)

# Hauptplot
plt.plot(df.index, recognition_rates, color='royalblue', linewidth=2, label='Recognition Rate')

# Bester k-Wert hervorheben
best_k = recognition_rates.idxmax()
best_val = recognition_rates.max()
plt.scatter(best_k, best_val, color='red', s=150, zorder=5,
            label=f'Best: k={best_k}, Rate={best_val}%', edgecolors='darkred', linewidths=2)

# Vertikale Linie beim besten k
plt.axvline(x=best_k, color='red', linestyle=':', alpha=0.5)

# Titel und Labels
plt.title('Hyperparameter Optimization: Recognition Rate vs. k (Unweighted k-NN)',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('k (Number of Neighbors)', fontsize=12)
plt.ylabel('Recognition Rate (%)', fontsize=12)

# Grid und Styling
plt.grid(True, linestyle='--', alpha=0.3)
plt.xlim(0, 310)
plt.ylim(recognition_rates.min() - 2, recognition_rates.max() + 2)

# Legende
plt.legend(loc='best', fontsize=10, framealpha=0.9)

# Layout optimieren
plt.tight_layout()
plt.show()

# Zusätzliche Information ausgeben
print(f"\n{'=' * 50}")
print(f"Hyperparameter Optimization Results:")
print(f"{'=' * 50}")
print(f"Best k: {best_k}") # 60
print(f"Best Recognition Rate: {best_val}%") # 77.6%
print(f"Worst Recognition Rate: {recognition_rates.min()}%") # 65.3%
print(f"Average Recognition Rate: {recognition_rates.mean():.1f}%") # 69.5%
print(f"{'=' * 50}\n")





