import numpy as np
import scipy as sp
import scipy.sparse as sps

def riemannian_metric(embedding: np.ndarray, laplacian: sps.csr_matrix):
    """
    Returns the dual riemannian metric (H), coordinate basis (U), and eigenvalues (S)
    for each inverse riemannian metric at each sample

    Adapted from https://github.com/mmp2/megaman/blob/master/megaman/geometry/rmetric.py,
    Algorithm 1 from https://arxiv.org/abs/1907.01651, and Algorithm 3 from https://arxiv.org/abs/1305.7255

    Parameters
    -----------------
    :param embedding: result embedding matrix
    :param laplacian: calculated Laplacian matrix

    Returns
    -----------------
    :returns H: Dual riemannian metric in the embedding space
    :returns U: Eigenvector basis of the tangent space of the embedding obtained from SVD of **H**
    :returns S: Eigenvalues corresponding to the eigenvectors
    """
    
    # Get estimated metric H_estimate
    dim = embedding.shape[1]
    N = embedding.shape[0]
    H_estimate = np.zeros((N, dim, dim)) # estimated dual riemannian metric

    for k in range(dim):
        for l in range(k, dim):
            ykl = embedding[:,k] * embedding[:,l]

            # Get estimated dual riemannian metric for all samples,
            # from Equation 21 https://arxiv.org/abs/1305.7255
            H_estimate[ :, k, l] = 0.5 * (laplacian.dot(ykl) - \
                       embedding[:, l] * laplacian.dot(embedding[:, k]) - \
                       embedding[:, k] * laplacian.dot(embedding[:, l]))
            
    for j in np.arange(dim - 1):
        for i in np.arange(j+1, dim):
            H_estimate[:, i, j] = H_estimate[:, j, i]

    # Compute H and G using SVD
    U, S, V = np.linalg.svd(H_estimate)

    return H_estimate, U, S
