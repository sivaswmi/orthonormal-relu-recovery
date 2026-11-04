import numpy as np
from scipy.linalg import qr
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA

def generate_ground_truth(d, k, m, n, p_noise=0, p_corrupt=0, seed=None):
    """
    Generate ground-truth two-layer ReLU network and Gaussian samples.

    Uses a local Generator, so the draw depends only on `seed` and not on
    global NumPy state -- this cell is safe to re-run in isolation.

    Can generate with noise and corruption

    Returns: V_star (k,d), U_star (m,k), X (d,n), A (m,n)
    """
    rng = np.random.default_rng(seed)

    G = rng.standard_normal((d, d))
    Q, R = qr(G)
    Q *= np.sign(np.diag(R))        # make Q Haar-distributed (LAPACK sign fix)
    V_star = Q[:k, :]               # orthonormal rows: V_star @ V_star.T = I_k

    U_star = rng.standard_normal((m, k))
    X = rng.standard_normal((d, n))
    A = U_star @ np.maximum(V_star @ X, 0)
    A += p_noise * rng.standard_normal((m, n))

    # corruption method is the same as shown in the Bulusu paper - N(5, 1)
    if p_corrupt > 0:
        n_bad = int(round(p_corrupt * n))
        idx = rng.choice(n, size=n_bad, replace=False)
        A[:, idx] = 5.0 + rng.standard_normal((m, n_bad))

    return V_star, U_star, X, A



