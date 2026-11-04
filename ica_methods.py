import numpy as np
from scipy.linalg import qr
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA
import gd_methods
from scipy.stats import norm



RELU_STD = np.sqrt(0.5 - 1 / (2 * np.pi))

def sketch_and_run_ICA(A, k, random_state=None):
    """
    Steps 1-3 of Algorithm 4.
    Returns:
        T      : (k, m) sketch matrix
        TU_hat : (k, k) estimated mixing matrix, estimate of T @ U*
                 (correct only up to unknown permutation Pi)
        F_hat :  (k, n) approximation of source signals up to ''
    """
    m, _ = A.shape
    rng = np.random.default_rng(random_state)
    T = rng.standard_normal((k, m))
    TA = T @ A

    ica = FastICA(n_components=k, random_state=random_state,
                   whiten='unit-variance', max_iter=5000)
    ica.fit(TA.T)

    TU_hat = ica.mixing_
    F_hat = np.linalg.solve(TU_hat, TA)

    # rectify sign issue here instead of doing at step 7 - makes things easier
    signs = np.sign(F_hat.mean(axis=1))
    F_hat *= signs[:, None]
    TU_hat = TU_hat * signs[None, :]

    # scale fix: map each row to the true rectified-Gaussian scale
    scale = F_hat.std(axis=1) / RELU_STD
    F_hat /= scale[:, None]
    TU_hat = TU_hat * scale[None, :]

    return T, TU_hat, F_hat



def restrict_to_ell(X, A, T, F_hat, ell):
    return X[:, :ell], A[:, :ell], T @ A[:, :ell], F_hat[:, :ell]

def estimate_noise_scale(F_bar):
    neg = F_bar[F_bar < 0]
    return float(np.sqrt(np.mean(neg**2)))

def heuristic_tau(F_bar):
    k, ell = F_bar.shape
    return estimate_noise_scale(F_bar) * norm.ppf(1 - 2 / (k * ell))

# Theorem 27 pins B_hat only up to an entry-wise positive diagonal, so Step 4's
# scalar tau is underdetermined as printed. We resolve the scale in
# sketch_and_run_ICA using the known rectified-Gaussian std, which puts tau on
# the same footing as the paper intends.

def threshold_f_hat(F_bar, tau=0.06):

    S = F_bar > tau
    return np.where(S, F_bar, 0.0), S


def solve_for_rows(TA_bar, S, method='svd'):
    # reconstruct the sparsity via finding perumations of A_bar's rows
    # follows from uniqueness theorem and shared rowspace by definition
    # of linear mapping

    k, ell = TA_bar.shape
    W = np.zeros((k, ell))
    diag = np.zeros(k)

    for j in range(k):
      off = ~S[j]                      # columns i in [ell] \ S_j
      M = TA_bar[:, off].T

      if method == 'svd':
        _, sv, Vt = np.linalg.svd(M, full_matrices=False)
        x = Vt[-1]
        diag[j] = sv[-1] / sv[-2]


      elif method == "lstsq":           # (n_off, k): one row per constraint
        best, best_res = None, np.inf
        for r in range(k):               # try each normalisation
            e_r = np.zeros(k); e_r[r] = 1.0
            lhs = np.vstack([M, e_r])                      # (n_off+1, k)
            rhs = np.concatenate([np.zeros(M.shape[0]), [1.0]])
            x_r, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            res = np.linalg.norm(M @ x_r)                      # how well it vanishes
            if res < best_res:
                best, best_res = x_r, res
        x, diag[j] = best, best_res

      else:
        raise ValueError(f"method must be 'svd' or 'lstsq', got {method!r}")



      w = x @ TA_bar
      W[j] = w * np.sign(w.mean())

    return W, diag


def recover_from_W(W, S, X_bar, X, A):
    k, ell = W.shape
    d = X_bar.shape[0]
    V_hat = np.zeros((k, d))
    n_sup = np.zeros(k, dtype=int)

    # recover V

    for i in range(k):
        S_i = S[i]
        n_sup[i] = S_i.sum()
        if n_sup[i] < d:
          raise ValueError(f"row {i}: |S_i|={n_sup[i]} < d={d}; increase ell")
        v, *_ = np.linalg.lstsq(X_bar[:, S_i].T, W[i, S_i], rcond=None)
        V_hat[i] = v / np.linalg.norm(v)

    # now recover U
    F = np.maximum(V_hat @ X, 0)
    U_hat, *_ = np.linalg.lstsq(F.T, A.T, rcond=None)

    return U_hat.T, V_hat, n_sup





def ica_exact_recovery_pipeline(A, X, k, ell=500, tau=None, random_state=0, method='svd', S_override=None):
    #end to end algo 4 implementation.
    T, TU_hat, F_hat = sketch_and_run_ICA(A, k, random_state)
    X_bar, A_bar, TA_bar, F_bar = restrict_to_ell(X, A, T, F_hat, ell)
    
    # recently added
    if tau is None:
        tau = heuristic_tau(F_bar)

    if S_override is None:
        F_thresh, S = threshold_f_hat(F_bar, tau)
    else:
        S = S_override[:, :ell]
    W, diag = solve_for_rows(TA_bar, S, method)
    U_hat, V_hat, n_sup = recover_from_W(W, S, X_bar, X, A)

    return U_hat, V_hat, {"T": T, "TU_hat": TU_hat, "F_hat": F_hat,
                          "S": S, "W": W, "diag": diag, "n_sup": n_sup, "tau": tau}

def sparsity_diagnosis(S, S_true, inv_perm):
   ell = S.shape[1]
   S = S[inv_perm]
   S_true = S_true[:, :ell]
   # minimal effect
   fn = (~S & S_true).sum()
   # large effect on error
   fp = (S & ~S_true).sum()
   total = S.size

   return {"err_rate": (fp + fn)/total,
           "fp_rate": fp/total,
           "fn_rate": fn/total}




def ica_exact_gd(A, X, k, ell=500, tau=None, random_state=0, method='svd', S_override=None, n_steps=2000, lr=0.01, seed=None, cosine=True):
    U_hat, V_hat, G = ica_exact_recovery_pipeline(A, X, k, ell=ell, tau=tau, random_state=random_state, method=method, S_override=S_override)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    U_int, V_int = U_hat.copy(), V_hat.copy()

    X_t = torch.tensor(X, dtype=torch.float64, device=dev)
    A_t = torch.tensor(A, dtype=torch.float64, device=dev)
    U_hat, V_hat, loss, best_step = gd_methods.train_gd(X_t, A_t, k, n_steps=n_steps, lr=lr, seed=seed, unit_norm_V=True,
              log_every=50, U_init=U_hat, V_init=V_hat, cosine=cosine)
    return U_hat, V_hat, loss, best_step, G, U_int, V_int



def spectral_initialisation(A, X, k):
   """
   Uses Stein's lemma
   spectral decomp on A to extract U and V initalisations - recover rowspace of V* and span of U*
   collect the k most prominent eigenvector/value combinations out of the spectral decomposition of A
   """
   n = X.shape[1]
   M = (A @ X.T) / n
   Um, s, Vt = np.linalg.svd(M, full_matrices=False)
   V_0 = Vt[:k]
   U_0 = Um[:, :k] * (2 * s[:k])
   return U_0, V_0





