import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Create outputs folder if not exists
os.makedirs("../outputs", exist_ok=True)

# ---------------------------
# Load Iris Dataset
# ---------------------------
iris = load_iris()
X = iris.data

# ---------------------------
# Scale Features
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Create Dendrogram
# ---------------------------
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram - Iris")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.savefig("../outputs/dendrogram.png")
plt.close()

# ---------------------------
# Apply Hierarchical Clustering
# ---------------------------
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X_scaled)

# ---------------------------
# Visualize Clusters
# ---------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.title("Hierarchical Clustering - Iris Dataset")
plt.xlabel("Sepal Length (scaled)")
plt.ylabel("Sepal Width (scaled)")
plt.savefig("../outputs/cluster_plot.png")
plt.close()

print("Clustering completed successfully!")
print("Outputs saved in the outputs folder.")