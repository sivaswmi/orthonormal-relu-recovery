import numpy as np
from scipy.linalg import qr, subspace_angles
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA


def align_and_score(U_hat, V_hat, U_star, V_star):
    """
    Aligns recovered (U_hat, V_hat) to ground truth (U_star, V_star) up to
    permutation and positive scaling ambiguity, then computes
    Frobenius norm parameter recovery error.
    """
    eps = 1e-12

    to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
    U_hat, V_hat = to_np(U_hat).astype(float), to_np(V_hat).astype(float)
    U_star, V_star = to_np(U_star), to_np(V_star)

    nU = np.linalg.norm(U_star, 'fro') + eps
    nV = np.linalg.norm(V_star, 'fro') + eps          # == sqrt(k) (orthonormal)

    k = V_star.shape[0]

    # scale: normalise rows of V to unit norm, push scale into U 
    s = np.linalg.norm(V_hat, axis=1) + eps
    V_g = V_hat / s[:, None]
    U_g = U_hat * s[None, :]

    # permutation resolving, Hungarian algo on joint squared distance
    dV = ((V_g[:, None, :] - V_star[None, :, :]) ** 2).sum(-1) / nV**2
    dU = ((U_g.T[:, None, :] - U_star.T[None, :, :]) ** 2).sum(-1) / nU**2
    _, col_ind = linear_sum_assignment(dV + dU)

    inv_perm = np.argsort(col_ind)
    V_aligned = V_g[inv_perm]
    U_aligned = U_g[:, inv_perm]

    return (U_aligned, V_aligned,
            np.linalg.norm(U_aligned - U_star, 'fro') / nU,
            np.linalg.norm(V_aligned - V_star, 'fro') / nV, inv_perm)



def subspace_metric(U_hat, V_hat, U_star, V_star):
    """
    Per-row angular error after permutation matching, in radians.

    Subspace angles are blind to rotation within the correct span. This matches rows to
    ground truth and reports the angle of each pair instead.
    """
    to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
    V_hat, V_star = to_np(V_hat).astype(float), to_np(V_star).astype(float)
    U_hat, U_star = to_np(U_hat).astype(float), to_np(U_star).astype(float)
    k = V_star.shape[0]

    vh = V_hat / np.linalg.norm(V_hat, axis=1, keepdims=True)
    vs = V_star / np.linalg.norm(V_star, axis=1, keepdims=True)
    Cv = np.abs(vh @ vs.T)                       # |cos| between every pair
    _, col = linear_sum_assignment(1 - Cv)
    V_ang = np.arccos(np.clip(Cv[np.arange(k), col], -1, 1))

    uh = U_hat / np.linalg.norm(U_hat, axis=0, keepdims=True)
    us = U_star / np.linalg.norm(U_star, axis=0, keepdims=True)
    Cu = np.abs(uh.T @ us)                       # U's columns index the hidden units
    U_ang = np.arccos(np.clip(np.diag(Cu[:, col]), -1, 1))

    return {"V_max": float(V_ang.max()), "U_max": float(U_ang.max()),
            "V_all": V_ang, "U_all": U_ang}