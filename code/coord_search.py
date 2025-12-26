# Taken from https://github.com/yuchaz/independent_coordinate_search/blob/master/ies_manifold/coord_search.py
from __future__ import division, print_function, absolute_import
import numpy as np
from itertools import combinations


def _comp_projected_volume(principal_space, proj_axis, intrinsic_dim,
                           embedding_dim, eigen_values=None, zeta=1):
    basis = principal_space[:, proj_axis, :min(intrinsic_dim, embedding_dim)]
    basis = basis / np.linalg.norm(basis, axis=1)[:, None, :]
    try:
        vol_sq = np.linalg.det(np.einsum(
            'ijk,ijl->ikl', basis, basis))
        parallelepipe_vol = np.sqrt(vol_sq)
    except Exception as e:
        print(vol_sq[vol_sq<0])
        parallelepipe_vol = np.sqrt(np.abs(vol_sq))

    regu_term = _calc_regularizer(eigen_values, proj_axis, zeta)
    return np.log(parallelepipe_vol) - regu_term


def _calc_regularizer(eigen_values, proj_axis, zeta=1):
    if eigen_values is None:
        return 0
    eigen_values = np.abs(eigen_values[proj_axis])
    regu_term = np.sum(eigen_values) * zeta
    return regu_term


def _projected_volume(principal_space, intrinsic_dim, embedding_dim=None,
                     eigen_values=None, zeta=1):
    candidate_dim = principal_space.shape[1]
    embedding_dim = intrinsic_dim if embedding_dim is None else embedding_dim

    all_axes = np.array(list(combinations(
        range(1, candidate_dim), embedding_dim-1)))
    all_axes = np.hstack([
        np.zeros((all_axes.shape[0], 1), dtype=all_axes.dtype), all_axes])

    proj_volume = []
    for proj_axis in all_axes:
        proj_vol = _comp_projected_volume(principal_space, proj_axis,
                                          intrinsic_dim, embedding_dim,
                                          eigen_values, zeta)
        proj_volume.append(proj_vol)

    proj_volume = np.array(proj_volume)
    return proj_volume, all_axes


def greedy_coordinate_search(principal_space, intrinsic_dim, eigen_values=None,
                             zeta=1):
    """
    Returns the optimal axes through independent eigendirection selection.
    Corresponds to Algorithm 2 of https://arxiv.org/abs/1907.01651 and adapted from 
    https://github.com/yuchaz/independent_coordinate_search/blob/master/ies_manifold/coord_search.py.

    Parameters
    -----------------
    :param principal_space: Shape:  (n_samples, n_dim, n_dim). Basis of the tangent space of the embedding.
    :param intrinsic_dim: Intrinsic dimension of the manifold
    :param eigen_values: Eigenvalues of the dual riemannian metric
    :param zeta: Regularization term

    Returns
    -----------------
    :returns opt_proj_axis: Indices of the optimal axis for a given intrinsic dimension
    """
    proj_vol, all_comb = _projected_volume(
        principal_space, intrinsic_dim, eigen_values=eigen_values, zeta=zeta)

    argmax_proj_vol = proj_vol.mean(1).argmax()
    opt_proj_axis = list(all_comb[argmax_proj_vol])

    return opt_proj_axis