"""
09-1_HierClust.py
    - In this module, we will apply agglomerative hierarchical clustering to a self-created synthetic data set.
    - We will optimize the number of clusters using the Silhouette Score as evaluation metric.
    - Finally, we will plot the dendogram of the hierarchical clustering using the scipy library.
"""

################ Preliminaries

from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import single, complete

# Remark:
# scipy is a library for scientific computing in Python.
#    - It is based on NumPy, and adds significant functionality, including functions for working with statistics, linear algebra, Fourier transforms, and more.
#    - It also provides many user-friendly and efficient numerical routines such as routines for numerical integration and optimization.
#    - We use scipy, because it is not (yet) possible to draw a dendogram from the scikit-learn model.
#    - Instead, we will use a scipy function to re-create the hierarchical clustering and then plot the dendogram from that.


################ Create a synthetic data set

# Create synthetic dataset with 4 "blobs" (clusters)
#
#    - n_samples=1000. Specify that we want to generate 1000 data objects.
#    - centers=4. Specify that we want 4 cluster centers (and thus 4 clusters)
#    - n_features=2. Specify number of features.
#    - random_state=42. Set a seed for reproducability.
#
#    Note:
#    - We intentionally only create a 2-dimensional data set.
#    - This way we can visualize it in a scatterplot, and later visually check our clustering results.
#    - This is useful in the beginning to get a feeling of what's happening.

X, y = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=42)

# Check the data structure in a scatterplot
plt.scatter(X[:, 0], X[:, 1]) # Plottet alle Datenpunkte mit Feature 0 auf der X-Achse und Feature 1 auf der Y-Achse
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

################ Preprocessing

# Note:
#    - It is not necessary to do a train/test split, since our data set does not contain examples of good clusters!
#    - Thus, a "test" set would not help us to test anything.
#    - Instead, the only way to evaluate the clustering quality is to use, e.g., the Dunn Index or the Silhouette Score,
#      to measure if the clustering matches the natural structure of the data well.


################ Fit (learn) a hierarchical clustering model

#   1. Initialize
#      Create an instance of the function AgglomerativeClustering() by specifying the following parameters:
#      - linkage="complete". Choose the compete linkage distance as a cluster distance.
#      - n_clusters=4. Specify that we want 4 clusters as a result.
#
#      Note 1:
#      - We choose 4 clusters, because we KNOW that the synthetic data set has 4 blobs (since we created it).
#      - This way we can easily check if the algorithm works well in finding the "natural structure" (the 4 blobs).
#      - In a real-world application, the correct number of clusters is typically not known in advance.
#
#      Note 2:
#      - The function AgglomerativeClustering() applies a hierarchical clustering approach.
#      - Specifying that we want 4 clusters "cuts" the resulting dendogram at an appropriate level to get 4 clusters.
#           -> 4 Exclusive Cluster extrahieren!
#      - Remember that the result is always an exclusive clustering (full and non-overlapping).
#
agg = AgglomerativeClustering(linkage="complete", n_clusters=4)

#   2. Fit and Predict
#
#      Note:
#      - Most traditional machine learning models in scikit-learn, such as linear regression and decision trees
#        have separate fit and predict methods. You first call fit(X_train, y_train) to train the model, then you
#        can call predict(X_test) to make predictions on new data.
#      - Yet, some models, particularly clustering algorithms provide a fit_predict method.
#        This method combines the fitting and predicting steps into one call.
#      fit:
#      - The fit step uses a hierarchical clustering algorithm to analyze the data structure in X and builds the hierarchical clustering model.
#      - It chooses an appropriate cutoff value to get an exclusive full clustering with 4 clusters.
#               Analysiert die Datenstruktur und baut das Hierarchical Clustering Model -> macht den Cut!
#      predict:
#      - The prediction step assigns each data point in X to one of the 4 clusters.
#      - The result is an array of cluster labels, where each label corresponds to the cluster assigned to each data point.
#         Ordnet die Datenpunkte in einer der 4 Cluster zu
#
#      IMPORTANT:
#      - For clustering, "prediction" is not really a prediction, but rather means ASSIGNING to each point its
#        cluster label. You can think of it as adding an additional column (feature) "cluster label" to your data set.
#      - The difference between PREDICTING and ASSIGNING is subtle but important:
#           - In supervised learning, we have an EXISTING attribute that we try to predict.
#           - In clustering, we are CREATING a new attribute (the cluster label) based on the data structure. -> predict = in Cluster zuordnen
#             (E.g. customers can be grouped this or that way, it's on us how to do that.)


assignment = agg.fit_predict(X)

################ Visually check the clustering result

# Add the cluster assignment as color to the scatter plot
#    - We see that our clustering indeed matches the 4 blobs nicely. It indeed found the natural structure of the data.
#    - This is not always the case in real-world applications, where the natural structure of the data is not that clear.
#    - You can run the clustering again with a different number of clusters (e.g., n_clusters=5) and check the result!
plt.scatter(X[:, 0], X[:, 1], c=assignment)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()


################ Hyper-Parameter Tuning with the Silhouette Score
# Anzahl Cluster = Hyper-Parameter
# Typically, the correct number of clusters in an application is not known: it is a hyperparameter.
#    - The following code fragment tests various hierarchical clusterings with different
#      numbers of clusters and calculates the silhouette score for each clustering.
#    - We see in the results that a clustering with 4 clusters indeed has the highest Silhouette Score.

# Find optimal number of clusters
for k in range(2, 8): #2 - 7
     agg = AgglomerativeClustering(linkage="complete", n_clusters=k)
     assignment = agg.fit_predict(X)
     score = silhouette_score(X, assignment)
     print("K =", k, "Silhouette Score =", score)

################ Plotting the Dendogram

# Unfortunately, we cannot directly plot a dendogram from our scikit-learn model! :-(
# Hopefully, this will be possible in future versions of scikit-learn.
# However, we can switch to the scipy library to do that:
#    - First, we re-create the hierarchical clustering in scipy using the function complete() from scipy.cluster.hierarchy.
#      The function complete() uses the complete-linkage distance (same as above).
#      Alternatively, we could also use single() from scipy.cluster.hierarchy to use the single-linkage distance.
#    - Then, we plot the dendogram using the function dendrogram().

#  The function complete() requires as input the data set X:
#  - It performs hierarchical clustering using the complete-linkage distance.
#  - It returns an array that specifies the cluster distances.
#    This array can then be used as input to the function dendrogram() to plot the dendogram.
#  - Notice that complete() performs hierarchical clustering on the entire data set X,
#    without specifying a number of clusters. Thus, the resulting dendogram shows the full hierarchy.
linkage_array = complete(X)
# Jede Zeile im Array repräsentiert einen Merge-Schritt


# Now we can plot the dendrogram using the function dendogram():
dendrogram(linkage_array)
plt.xlabel("Data Object")
plt.ylabel("Cluster distance")
plt.show()

# The dendogram plot above highlights 3 clusters by default.
#    - However, we KNOW that our data set has 4 clusters (since we created it that way).
#      Thus, we want to see all 4 clusters in the dendogram.
#    - To highlight the 4 clusters visually, we can use a threshold of 10 for the cluster distance.
#      This means that all subclusters below height 10 will be colored differently.
#      Thus, we call dendrogram() again with the parameter color_threshold=10:
dendrogram(linkage_array, color_threshold=10)
plt.xlabel("Data Object")
plt.ylabel("Cluster Distance (Complete Linkage)")
plt.show()
