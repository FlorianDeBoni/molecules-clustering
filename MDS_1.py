import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import json

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
with open("output/clustering_results.json", "r") as f:
    data = json.load(f)

snapshots = data["snapshots"]
clusters = np.array([snap["cluster"] for snap in snapshots])
centroids_idx = np.array(data["centroids"])

D = np.loadtxt("output/rmsd_matrix.csv", delimiter=",")

# ------------------------------------------------------------------
# MDS
# ------------------------------------------------------------------
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0, n_jobs=-1)
X_mds = mds.fit_transform(D)

# ------------------------------------------------------------------
# Colors
# ------------------------------------------------------------------
unique_clusters = np.unique(clusters)
K = len(unique_clusters)

cmap = plt.get_cmap("tab10", K)
colors = cmap(clusters)

# ------------------------------------------------------------------
# Plot points
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))

plt.scatter(
    X_mds[:, 0],
    X_mds[:, 1],
    c=colors,
    s=30,
    alpha=0.7
)

# ------------------------------------------------------------------
# Plot centroids with SAME color
# ------------------------------------------------------------------
centroid_colors = cmap(clusters[centroids_idx])

plt.scatter(
    X_mds[centroids_idx, 0],
    X_mds[centroids_idx, 1],
    c=centroid_colors,
    s=270,
    marker="X",
    edgecolors="black",
    linewidths=1.5,
    label="Centroids"
)

# ------------------------------------------------------------------
# Colorbar propre
# ------------------------------------------------------------------
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=K-1))
sm.set_array([])

cbar = plt.colorbar(sm, ticks=range(K))
cbar.set_label("Cluster ID")

# ------------------------------------------------------------------
# Final touches
# ------------------------------------------------------------------
plt.title("Projection MDS",fontsize=30)
plt.xlabel("Dim 1",fontsize=25)
plt.ylabel("Dim 2",fontsize=25)
plt.legend()

plt.tight_layout()
plt.show()