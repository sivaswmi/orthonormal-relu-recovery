import numpy as np
import pandas as pd
from scipy.linalg import qr
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA
from scipy.linalg import null_space

def forward_and_loss(U, V, X, A):
    A_pred = U @ torch.relu(V @ X)
    # normalise by number of elements for per-element loss
    return torch.sum((A - A_pred)**2)


def train_gd(X_t, A_t, k, n_steps=2000, lr=0.01, seed=None, unit_norm_V=True,
              log_every=50, U_init=None, V_init=None, cosine=True, hess_info=False, hess_every=50):
    """
    Full-batch Adam baseline for A = U f(V X).

    Uses a local torch.Generator so the init depends only on `seed`,
    not on global torch state. Runs on whatever device X_t lives on.
    """
    dev = X_t.device


    g = torch.Generator()                            
    if seed is not None:
        g.manual_seed(seed)

    m, d = A_t.shape[0], X_t.shape[0]

    if U_init is None and V_init is None:
        U_0 = torch.randn(m, k, generator=g, dtype=torch.float64)
        V_0 = torch.randn(k, d, generator=g, dtype=torch.float64)
        if unit_norm_V:
            V_0 = V_0 / V_0.norm(dim=1, keepdim=True)
    else:
        U_0 = torch.as_tensor(U_init, dtype=torch.float64)
        V_0 = torch.as_tensor(V_init, dtype=torch.float64)

    U_hat = U_0.to(dev).requires_grad_(True)
    V_hat = V_0.to(dev).requires_grad_(True)

    optimizer = torch.optim.Adam([U_hat, V_hat], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps) if cosine else None

    best = (float('inf'), None, None, -1)
    losses, steps = [], []

    hess_log = []
    if hess_info:
        Xn = X_t.detach().cpu().numpy()
        An = A_t.detach().cpu().numpy()
        Xnorm = np.linalg.norm(Xn, 2)
    
    

    for step in range(n_steps):
        optimizer.zero_grad()
        loss = forward_and_loss(U_hat, V_hat, X_t, A_t)
        loss.backward()
        optimizer.step()

        if sched: sched.step()
        l = loss.item()
        if l < best[0]:
            best = (l, U_hat.detach().clone(), V_hat.detach().clone(), step)

        if step % log_every == 0:
            losses.append(loss.detach())             # stays on device, no sync
            steps.append(step)

        if hess_info and step % hess_every == 0:
            hd = hessian_diagnostics(U_hat.detach().cpu().numpy(),
                                     V_hat.detach().cpu().numpy(), Xn, An, Xnorm)
            hd["step"] = step
            hd["loss"] = l
            hess_log.append(hd)



    losses = (torch.stack(losses) / A_t.numel()).cpu().numpy()   # normalise
    best_loss, U_best, V_best, best_step = best

    if not hess_info:
        return U_best.cpu(), V_best.cpu(), (steps, losses), best_step
    
    
    return U_best.cpu(), V_best.cpu(), (steps, losses), best_step, pd.DataFrame(hess_log)


def K(a, b):                                       
    Km = np.zeros((a*b, a*b))
    for i in range(a):
        for j in range(b):
            Km[i*b + j, j*a + i] = 1
    return Km



def symmetry_basis(U, V):
    """
    Orthonormal basis N of the complement of T_sym at (U, V).
    T_sym is spanned by z_i = (U diag(e_i), -diag(e_i) V), i = 1..k,
    vectorised column-major to match the columns of J = [J_U, J_V].
    """
    m_, k_ = U.shape
    vec = lambda P: P.reshape(-1, order='F')
    Z = []
    for i in range(k_):
        dU = np.zeros_like(U); dU[:, i] = U[:, i]
        dV = np.zeros_like(V); dV[i, :] = -V[i, :]
        Z.append(np.concatenate([vec(dU), vec(dV)]))
    Z = np.array(Z).T                     # (p, k), columns span T_sym
    return null_space(Z.T)  

def _vec(P):
    """Column vectorisation matching the Kronecker identities."""
    return P.reshape(-1, order='F')


def compute_jacobian(U, V, X):
    """J = [J_U, J_V], the Jacobian of vec(R) w.r.t. z = [vec U; vec V].
    Shape (mn, mk + kd). Also returns F = relu(VX) and the activation mask dvec."""
    m, k = U.shape
    d, n = X.shape
    F = np.maximum(V @ X, 0)
    dvec = _vec((V @ X) > 0).astype(float)
    J_U = np.kron(F.T, np.eye(m))
    J_V = (np.kron(np.eye(n), U) * dvec) @ np.kron(X.T, np.eye(k))
    return np.hstack([J_U, J_V]), F, dvec


def compute_S(U, V, X, A, F=None, dvec=None):
    """Residual term S of H = 2 J^T J + S. Block off-diagonal, proportional to R."""
    m, k = U.shape
    d, n = X.shape
    if F is None:
        F = np.maximum(V @ X, 0)
    if dvec is None:
        dvec = _vec((V @ X) > 0).astype(float)
    R = U @ F - A
    p, q = m * k, k * d
    S_UV = 2 * ((np.kron(np.eye(k), R) @ K(k, n)) * dvec) @ np.kron(X.T, np.eye(k))
    S = np.zeros((p + q, p + q))
    S[:p, p:] = S_UV
    S[p:, :p] = S_UV.T
    return S, R


def compute_hessian(U, V, X, A):
    """Closed-form Hessian H = 2 J^T J + S of f(U,V) = ||U relu(VX) - A||_F^2.
    Dense Kronecker construction: small configurations only (e.g. (8,4,6), n <= 2000).
    Returns a dict with H, JtJ, S, J, R."""
    J, F, dvec = compute_jacobian(U, V, X)
    S, R = compute_S(U, V, X, A, F, dvec)
    JtJ = J.T @ J
    return {"H": 2 * JtJ + S, "JtJ": JtJ, "S": S, "J": J, "R": R}


def lambda_perp(M, N):
    """Smallest eigenvalue of M restricted to the complement of T_sym (columns of N)."""
    return np.linalg.eigvalsh(N.T @ M @ N)[0]


def hessian_diagnostics(U, V, X, A, Xnorm=None):
    """Certificate diagnostics at (U, V): residual, lambda_perp of J^T J and H,
    ||S|| and its bound, and whether Corollary 1's certificate holds."""
    h = compute_hessian(U, V, X, A)
    N = symmetry_basis(U, V)
    if Xnorm is None:
        Xnorm = np.linalg.norm(X, 2)

    resid = np.linalg.norm(h["R"])                    
    # ||R||_F

    lamGN = lambda_perp(h["JtJ"], N)                  
    # lambda_perp(J^T J)

    lam_perp_H = lambda_perp(h["H"], N)               
    # true curvature off T_sym

    S_full = np.linalg.norm(h["S"], 2)                
    # ||S||_2

    S_proj = np.linalg.norm(N.T @ h["S"] @ N, 2)      
    # ||N^T S N||_2 (tighter)

    S_bound = 2 * resid * Xnorm                       
    # Lemma 4 bound

    return {"resid":      float(resid),
            "lamGN":      float(lamGN),
            "rhs":        float(lamGN / Xnorm),
            "cert":       bool(resid < (lamGN / Xnorm)),
            "S_full":     float(S_full),
            "S_proj":     float(S_proj),
            "S_bound":    float(S_bound),
            "weyl":       float(2 * lamGN - S_full),
            "lam_perp_H": float(lam_perp_H),
            "lam_min_H":  float(np.linalg.eigvalsh(h["H"])[0])}   # unprojected, comparison only


def first_stable_step(df, col, cond):
    """First logged step after which cond(df[col]) holds for every later log."""
    ok = cond(df[col].to_numpy())
    for i in range(len(ok)):
        if ok[i:].all():
            return int(df["step"].iloc[i])
    return None



def loss_graph(steps, losses):
    plt.plot(steps, losses)
    plt.yscale('log')
    plt.xlabel('step'); plt.ylabel('loss per entry')
    plt.title('GD training loss'); plt.show()
