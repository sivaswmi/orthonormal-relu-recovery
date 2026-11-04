import scoring
import gd_methods
import ica_methods
import ground_truth
import torch
import numpy as np
import pandas as pd




# contains all run methods for each created
# intended to allow for collating results, and future grid search. 
# outputs dataframes

"""
Params needed for run_all:
n, d, k, m, p_error, p_corrupt, ell, tau, lr, cosine, n_steps

output factors:
method, n, d, k, m, p_error, p_corrupt, ell, tau, lr, cosine, n_steps, U rec error, V rec error, best iterate, Sparsity %, FN %, subspace angle
"""

def run_ica_algo4(cfg, gt, ell=500, tau=None, random_state=0, method='svd'):
    V_star, U_star, X, A = gt

   
    try:
        U, V, misc = ica_methods.ica_exact_recovery_pipeline(
            A, X, cfg["k"], ell=ell, tau=tau, random_state=random_state, method=method)
    except ValueError as e:
        return {"method": "ICA Algo4", "ell": ell, "tau": tau, "U_err": np.nan,
                "V_err": np.nan, "failed": str(e)}
    
    _, _, eU, eV, perm = scoring.align_and_score(U, V, U_star, V_star)

    S_true = np.maximum(V_star @ X[:, :ell], 0) > 0
    sp = ica_methods.sparsity_diagnosis(misc["S"], S_true, perm)

    return {"method": "ICA Algo4", "ell": ell, "tau": misc["tau"],
            "solve_method": method, "random_state": random_state,
            "U_err": eU, "V_err": eV,
            "sp_err": sp["err_rate"], "sp_fn": sp["fn_rate"],
            "diag_max": float(misc["diag"].max())}



def run_gd_baseline(cfg, gt, n_steps=2000, lr=0.01, cosine=True):
    """cfg: dict of problem params. gt: (V_star, U_star, X, A). Returns a result dict."""
    V_star, U_star, X, A = gt
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_t = torch.tensor(X, dtype=torch.float64, device=dev)
    A_t = torch.tensor(A, dtype=torch.float64, device=dev)

    U, V, loss, best_step = gd_methods.train_gd(X_t, A_t, cfg["k"],
                                            n_steps=n_steps, lr=lr, cosine=cosine)
    _, _, eU, eV, _ = scoring.align_and_score(U, V, U_star, V_star)

    return {"method": "GD Baseline", "n_steps": n_steps, "lr": lr, "cosine": cosine,
            "U_err": eU, "V_err": eV, "best_step": best_step}




def run_ica_exact_gd(cfg, gt, ell=500, tau=None, random_state=0, method='svd',
                     n_steps=2000, lr=0.01, seed=None, cosine=True):
    V_star, U_star, X, A = gt
    U, V, loss, best_step, misc, U_int, V_int = ica_methods.ica_exact_gd(
        A, X, cfg["k"], ell=ell, tau=tau, random_state=random_state,
        method=method, n_steps=n_steps, lr=lr, seed=seed, cosine=cosine)
    _, _, eU, eV, _ = scoring.align_and_score(U, V, U_star, V_star)
    sub = scoring.subspace_metric(U_int, V_int, U_star, V_star)
    return {"method": "Algo4+GD", "ell": ell, "tau": misc["tau"],
            "U_err": eU, "V_err": eV, "best_step": best_step,
            "init_V_angle": float(sub["V_max"]), "init_U_angle": float(sub["U_max"])}


def run_spectral_gd(cfg, gt, n_steps=2000, lr=0.01, seed=None, cosine=True):
    V_star, U_star, X, A = gt
    U_0, V_0 = ica_methods.spectral_initialisation(A, X, cfg["k"])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_t = torch.tensor(X, dtype=torch.float64, device=dev)
    A_t = torch.tensor(A, dtype=torch.float64, device=dev)
    U, V, loss, best_step = gd_methods.train_gd(
        X_t, A_t, cfg["k"], n_steps=n_steps, lr=lr, seed=seed,
        U_init=U_0, V_init=V_0, cosine=cosine)
    _, _, eU, eV, _ = scoring.align_and_score(U, V, U_star, V_star)
    sub = scoring.subspace_metric(U_0, V_0, U_star, V_star)
    return {"method": "Spectral+GD",
            "U_err": eU, "V_err": eV, "best_step": best_step,
            "init_V_angle": float(sub["V_max"]), "init_U_angle": float(sub["U_max"])}



PROBLEM_KEYS = {"d", "k", "m", "n", "p_noise", "p_corrupt", "seed"}
def run_all(configs, methods, out="results.csv"):
    """
    configs : list of dicts. Problem params (d,k,m,n,p_noise,p_corrupt,seed) go to
              the generator; anything else is carried through for the methods to
              read and lands in the results row.
    methods : dict of name -> callable(cfg, gt) returning a result dict.
    """
    
    rows = []
    for cfg in configs:
        prob = {k: v for k, v in cfg.items() if k in PROBLEM_KEYS}
        gt = ground_truth.generate_ground_truth(**prob)
        for name, fn in methods.items():
            try:
                r = fn(cfg, gt)
            except Exception as e:
                r = {"method": name, "error": repr(e)}
            rows.append({**cfg, **r, "method": name})
        pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)
