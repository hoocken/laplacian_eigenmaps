import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from astroML.datasets import sdss_corrected_spectra

def local_pca(data: np.ndarray, k: int) -> float:
    """
    Generates a dimension estimation using local PCA for a given dataset.
    
    Parameters
    ------------
    :param data: Dataset of the manifold
    :param k: Amount of neighbors for k-Nearest Neighbors


    Returns
    ------------
    :returns d_estimate: Estimation of the dimension
    """
    # For a point with its neighborhood, generate PCA
    nbr_graph = NearestNeighbors(n_neighbors=k).fit(data)
    _, indices = nbr_graph.kneighbors(data)

    sum_d_estimates = 0
    N = data.shape[0]
    for i in range(N): 
        # Get non-zero entries for a row
        neighbors = indices[i]

        # Add own node to neighborhood
        local_data = np.insert(data[neighbors], 0, data[i], axis=0)

        fitted_PCA = PCA().fit(local_data)
        sum_d_estimates += np.sum(fitted_PCA.explained_variance_ > 0.05 * fitted_PCA.explained_variance_[0])

        if (i % 100 == 0 and i != 0):
            print(f"Estimated d with i = {i} is d = {sum_d_estimates / i}")

    return sum_d_estimates / N