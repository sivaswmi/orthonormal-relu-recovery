# Recovering Two-Layer ReLU Networks: ICA vs Gradient Descent

Code for my COMP30013 research thesis (University of Melbourne, 2026), supervised by Dr Jean Honorio.
The thesis is included as [`thesis.pdf`](thesis.pdf).

We study recovery of the weights of A = U* ReLU(V* X) with orthonormal V* from Gaussian inputs, comparing the
ICA-based exact recovery algorithm of Bakshi, Jayaram & Woodruff (COLT 2019) with gradient descent:

- a finite-sample implementation of their orthonormal-V* algorithm, and a diagnosis of where its error comes from (the thresholding step)
- a ground-truth-free threshold rule, tau* = sigma_hat * Phi^{-1}(1 - 2/(k*ell))
- ICA as an initialiser for gradient descent
- the closed-form Hessian of the loss, and a computable curvature certificate.

## Setup

Tested with Python 3.11. CPU is sufficient.

    pip install -r requirements.txt jupyter
    jupyter notebook reproduce.ipynb

## Files

| File | Contents |
|---|---|
| `ground_truth.py` | Data generation (orthonormal V*, Gaussian U* and X, optional noise and contamination) |
| `ica_methods.py` | Algorithm 2 of the thesis: sketching, FastICA, thresholding, null-space and least-squares steps; spectral initialisation |
| `gd_methods.py` | Full-batch Adam baseline; closed-form Hessian, symmetry basis and certificate diagnostics |
| `scoring.py` | Alignment up to permutation and positive scaling; recovery errors; per-unit angles |
| `exp_run.py` | Helpers for running method/configuration sweeps into a DataFrame |
| `reproduce.ipynb` | Reproduces the results below |

## Thesis results → notebook

| Thesis | Notebook section |
|---|---|
| Table 2 (oracle support vs end-to-end) | Oracle sparsity check; end-to-end Algorithm 4 |
| Table 3 (support error vs tau) | ICA error vs thresholding bottleneck |
| Figures 1–2 (tau sweep, heuristic tau*) | Tau sweep plots |
| Table 4 (asymmetric error costs) | Asymmetric cost of support errors |
| Figure 3, Table 6 (GD from random / spectral / ICA) | GD baseline; ICA initialisation; spectral initialisation |
| Hessian verification (§3.3.1) | Closed-form Hessian vs autodiff |
| Table 8, §3.3.4 (curvature certificate) | S-ceiling sweep at (m,k,d) = (8,4,6); ICA certificate check |
| Table 5 (noise and contamination) | Sweep with `exp_run` |

## Notes

- Runs use fixed seeds, but FastICA results can still vary slightly across library versions.
- The certificate sweep averages over 5 random perturbation directions per radius; `cert_frac` is the fraction of directions certified.
- The thesis reports an oracle-tuned threshold tau = 0.212, which minimises support error. Recovery error is minimised at a smaller tau, where false positives first vanish; `tau=None` uses the heuristic tau*.
- Gradient-descent step counts depend on the cosine-annealing schedule length (`n_steps`) - compare initialisations at the same schedule. The code here differs from what is presented in the thesis - playing around with step count along with
cosine annealing reveals that ICA initialisation is much more efficient than what is reported, with up to 10x faster covnvergence as opposed to the 4x that was reported.
- GD cells take several minutes on CPU, the noise/contamination sweep takes longer.
- The heuristic τ assumes the only noise is ICA estimation error. Under additive noise it’s inflated. This results in failures for the final sweep cell - fix tau = 0.19 to help fix this, for contamination runs.
- Algo4+GD’s best_step of about 1,800–2,000 with n_steps=2000 means it’s running against the end of its schedule. It’s not a fair “steps to converge” number to set against GD’s 8,000. Only compare final errors from this sweep, not step counts.

## Citation

    Sivaram Swaminathan, Jean Honorio. Recovering Two-Layer ReLU Networks: Exact Recovery, ICA
    Initialisation and Local Geometry. COMP30013 research thesis, University of
    Melbourne, 2026.

## License

Code: MIT (see `LICENSE`).