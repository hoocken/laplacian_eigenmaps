import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

# ----------------------------
# Styling
# ----------------------------
plt.style.use("dark_background")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 11})

# ----------------------------
# Data: Anthony's Swiss roll generator
# ----------------------------
"""
rng = np.random.default_rng(42)

length_phi = 15
length_Z = 15
sigma = 2
N = 2000

phi = length_phi * rng.random(N)
xi = rng.random(N)
Z = length_Z * rng.random(N)

r = (phi + sigma * xi) / 6.0
Xc = r * np.sin(phi)
Yc = r * np.cos(phi)

X = np.column_stack([Xc, Yc, Z])

# Use phi as the intrinsic color parameter (analogous to sklearn's t)
t = phi
"""

#The ‘rectangle-like’ shape depends heavily on the sampling/parameterization of the Swiss roll and on which
#eigenvectors you plot. Laplacian Eigenmaps is designed to preserve local neighborhood structure via smooth 
#Laplacian eigenfunctions, not to recover a globally flat parametrization.

# ----------------------------
# Data
# ----------------------------

X, t = make_swiss_roll(n_samples=2000, noise=0.05, random_state=42)

X_pca = PCA(n_components=2).fit_transform(X)

# ----------------------------
# Laplacian Eigenmaps (Belkin–Niyogi style)
# ----------------------------
k = 8

# Sparse kNN distance graph (only edges), exclude self-neighbor
G = kneighbors_graph(X, n_neighbors=k, mode="distance", include_self=False)

# Choose bandwidth s ~ median kNN distance (robust)
s = np.median(G.data)

# Heat kernel weights on edges
G.data = np.exp(-(G.data**2) / (2 * s**2))

# Symmetrize W (undirected graph)
W = 0.5 * (G + G.T)

# Degree and Laplacian (sparse)
d = np.asarray(W.sum(axis=1)).ravel()
D = sps.diags(d)
L = D - W

# Generalized eigenproblem: (D - W) v = λ D v
# Compute smallest eigenpairs: need 1 extra to drop the trivial eigenvector
vals, vecs = eigsh(L, k=3, M=D, which="SM")  # SM = smallest magnitude
idx = np.argsort(vals)
vals, vecs = vals[idx], vecs[:, idx]

X_le = vecs[:, 1:3]  # drop trivial constant eigenvector

# ----------------------------
# Plot (your 1x3 layout)
# ----------------------------
fig = plt.figure(figsize=(15, 4.5))

ax1 = fig.add_subplot(1, 3, 1, projection="3d")
sc = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap="Spectral",
                 s=5, alpha=0.85, marker=".", linewidths=0)
ax1.set_title("Swiss Roll (3D)")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
ax1.view_init(elev=12, azim=120)
ax1.xaxis.pane.fill = False; ax1.yaxis.pane.fill = False; ax1.zaxis.pane.fill = False
ax1.grid(False)

ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=t, cmap="Spectral",
            s=5, alpha=0.85, marker=".", linewidths=0)
ax2.set_title("PCA")
ax2.set_xlabel("PC 1"); ax2.set_ylabel("PC 2")
ax2.set_aspect("equal", adjustable="box")

ax3 = fig.add_subplot(1, 3, 3)
ax3.scatter(X_le[:, 0], X_le[:, 1], c=t, cmap="Spectral",
            s=5, alpha=0.85, marker=".", linewidths=0)
ax3.set_title("Laplacian Eigenmaps")
ax3.set_xlabel(r"$\phi_1$"); ax3.set_ylabel(r"$\phi_2$")
ax3.set_aspect("equal", adjustable="box")

fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.22, wspace=0.35)
cax = fig.add_axes([0.25, 0.06, 0.50, 0.035])
fig.colorbar(sc, cax=cax, orientation="horizontal").set_label("Swiss roll parameter $t$")

plt.savefig("swiss_roll_pca_le_comparison.png", bbox_inches="tight", pad_inches=0.15)
plt.show()
