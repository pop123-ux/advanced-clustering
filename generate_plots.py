"""Generate portfolio plots for the README:
  - algorithm comparison on make_moons (KMeans / DBSCAN / Agglo-Ward / Agglo-Single / Spectral)
  - single-linkage dendrogram
  - DBSCAN eps-sensitivity sweep

Run from the repo root:
    python generate_plots.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import (AgglomerativeClustering, DBSCAN, KMeans,
                             SpectralClustering)
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
IMG = HERE / "images"
IMG.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Generating make_moons dataset...")
X, y = make_moons(n_samples=600, noise=0.08, random_state=RANDOM_STATE)
X = StandardScaler().fit_transform(X)

# --- Algorithm comparison ---
print("Running all 5 algorithms on make_moons...")
labels_km = KMeans(n_clusters=2, n_init=20, random_state=RANDOM_STATE).fit_predict(X)
labels_db = DBSCAN(eps=0.25, min_samples=6).fit_predict(X)
labels_aw = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(X)
labels_as = AgglomerativeClustering(n_clusters=2, linkage="single", metric="euclidean").fit_predict(X)
labels_sp = SpectralClustering(
    n_clusters=2, affinity="nearest_neighbors", n_neighbors=12,
    assign_labels="kmeans", random_state=RANDOM_STATE,
).fit_predict(X)

results = [
    ("Ground truth", y),
    (f"K-Means (ARI={adjusted_rand_score(y, labels_km):.3f})", labels_km),
    (f"DBSCAN (ARI={adjusted_rand_score(y, labels_db):.3f})", labels_db),
    (f"Agglomerative – Ward (ARI={adjusted_rand_score(y, labels_aw):.3f})", labels_aw),
    (f"Agglomerative – Single (ARI={adjusted_rand_score(y, labels_as):.3f})", labels_as),
    (f"Spectral (ARI={adjusted_rand_score(y, labels_sp):.3f})", labels_sp),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (title, lab) in zip(axes.flat, results):
    ax.scatter(X[:, 0], X[:, 1], c=lab, s=20, cmap="tab10")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Algorithm comparison on make_moons (5 methods + ground truth)", fontsize=14)
plt.tight_layout()
plt.savefig(IMG / "comparison_moons.png", dpi=140)
plt.close()
print(f"Saved {IMG / 'comparison_moons.png'}")

# --- Dendrogram (single-linkage on a sub-sample for legibility) ---
print("Building dendrogram...")
idx = np.random.choice(len(X), size=80, replace=False)
Z = linkage(X[idx], method="single", metric="euclidean")

fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z, no_labels=True, ax=ax, color_threshold=0.5)
ax.set_title("Single-linkage dendrogram on an 80-point sub-sample of make_moons")
ax.set_xlabel("Observations")
ax.set_ylabel("Fusion distance")
plt.tight_layout()
plt.savefig(IMG / "dendrogram.png", dpi=140)
plt.close()
print(f"Saved {IMG / 'dendrogram.png'}")

# --- DBSCAN eps sensitivity ---
print("DBSCAN eps sensitivity sweep...")
eps_values = [0.12, 0.16, 0.22, 0.30]
fig, axes = plt.subplots(1, len(eps_values), figsize=(16, 4))
for ax, eps in zip(axes, eps_values):
    lab = DBSCAN(eps=eps, min_samples=6).fit_predict(X)
    ari = adjusted_rand_score(y, lab)
    ax.scatter(X[:, 0], X[:, 1], c=lab, s=18, cmap="tab10")
    ax.set_title(f"eps = {eps}\nARI = {ari:.3f}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("DBSCAN parameter sensitivity — eps sweep on make_moons", fontsize=13)
plt.tight_layout()
plt.savefig(IMG / "sensitivity.png", dpi=140)
plt.close()
print(f"Saved {IMG / 'sensitivity.png'}")

print("\nAll plots generated successfully.")
