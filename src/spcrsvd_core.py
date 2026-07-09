
import numpy as np
from numpy.linalg import svd, solve, norm


# ── helpers ──────────────────────────────────────────────────

def soft(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))

def proj_stiefel(V):
    U, _, Vt = svd(V, full_matrices=False)
    return U @ Vt

def sinThetaF(U1, U2):
    P1 = U1 @ U1.T
    P2 = U2 @ U2.T
    return np.sqrt(0.5 * np.sum((P1 - P2) ** 2))

def standardize_from_train(Xtr, Xother):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, ddof=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (Xtr - mu) / sd, (Xother - mu) / sd

def center_y_from_train(ytr, yother):
    mu = ytr.mean()
    return ytr - mu, yother - mu

def toeplitz_cov(p, rho):
    idx = np.arange(p)
    return rho ** np.abs(np.subtract.outer(idx, idx))

def spcr_objective(X, y, V, Z, beta0, beta, w, lamV, lamB):
    n = X.shape[0]
    reg = 0.5 / n * np.sum((y - beta0 - X @ V @ beta) ** 2)
    pca = 0.5 * w / n * np.sum((X - Z @ V.T) ** 2)
    penV = lamV * np.sum(np.abs(V))
    penB = lamB * np.sum(np.abs(beta))
    return reg + pca + penV + penB


# ── DGP ──────────────────────────────────────────────────────

def generate_sparse_block_orthonormal(p, k, support_size, rng):
    V = np.zeros((p, k))
    perm = rng.permutation(p)
    start = 0
    for j in range(k):
        supp = perm[start:start + support_size]
        start += support_size
        v = rng.normal(size=support_size)
        v /= max(norm(v), 1e-12)
        V[supp, j] = v
    return proj_stiefel(V)

def generate_dgp(
    n, p, k,
    support_size=12,
    nuisance_rank=4,
    nuisance_strength=14.0,
    signal_strength=1.5,
    rho_noise=0.30,
    noise_x=0.35,
    noise_y=0.45,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()
    total_rank = nuisance_rank + k
    B_all = generate_sparse_block_orthonormal(p, total_rank, support_size, rng)
    B_nuis = B_all[:, :nuisance_rank]
    B_sig  = B_all[:, nuisance_rank:nuisance_rank + k]
    F_nuis = rng.normal(size=(n, nuisance_rank)) * np.sqrt(nuisance_strength)
    F_sig  = rng.normal(size=(n, k)) * np.sqrt(signal_strength)
    E = rng.multivariate_normal(np.zeros(p), toeplitz_cov(p, rho_noise), size=n)
    E *= noise_x
    X = F_nuis @ B_nuis.T + F_sig @ B_sig.T + E
    gamma = np.linspace(2.4, 1.4, k)
    y = F_sig @ gamma + noise_y * rng.normal(size=n)
    return X, y, B_sig


# ── Sylvester solvers ────────────────────────────────────────

def solve_V1_sylvester(XtX_over_n, beta, rho2, RHS):
    p, k = RHS.shape
    bnorm2 = float(beta @ beta)
    if bnorm2 < 1e-14:
        return RHS / rho2
    e1 = beta / np.sqrt(bnorm2)
    Q = np.eye(k); Q[:, 0] = e1
    Q, _ = np.linalg.qr(Q)
    RHS_Q = RHS @ Q
    V1_Q = np.zeros_like(RHS_Q)
    A0 = bnorm2 * XtX_over_n + rho2 * np.eye(p)
    V1_Q[:, 0] = solve(A0, RHS_Q[:, 0])
    for j in range(1, k):
        V1_Q[:, j] = RHS_Q[:, j] / rho2
    return V1_Q @ Q.T

def solve_V1_sylvester_woodbury(X, beta, rho2, RHS):
    n, p = X.shape
    _, k = RHS.shape
    bnorm2 = float(beta @ beta)
    if bnorm2 < 1e-14:
        return RHS / rho2
    e1 = beta / np.sqrt(bnorm2)
    Q = np.eye(k); Q[:, 0] = e1
    Q, _ = np.linalg.qr(Q)
    RHS_Q = RHS @ Q
    V1_Q = np.zeros_like(RHS_Q)
    c = bnorm2 / n
    rhs0 = RHS_Q[:, 0]
    Xrhs = X @ rhs0
    mid = solve(np.eye(n) + (c / rho2) * (X @ X.T), Xrhs)
    V1_Q[:, 0] = rhs0 / rho2 - (c / (rho2 ** 2)) * (X.T @ mid)
    for j in range(1, k):
        V1_Q[:, j] = RHS_Q[:, j] / rho2
    return V1_Q @ Q.T


# ── feasible projection ─────────────────────────────────────

def row_support_mask(A, s_keep):
    row_norms = np.sqrt(np.sum(A ** 2, axis=1))
    idx = np.argsort(-row_norms)[:s_keep]
    mask = np.zeros(A.shape[0], dtype=bool)
    mask[idx] = True
    return mask

def refit_beta(X, y, V, ridge=1e-6):
    XV = X @ V
    k = XV.shape[1]
    beta = solve((XV.T @ XV) / X.shape[0] + ridge * np.eye(k),
                 (XV.T @ y) / X.shape[0])
    beta0 = float(np.mean(y - XV @ beta))
    return beta0, beta

def make_feasible_state(X, y, V, V0, V1, beta, beta_clone,
                        w, lamV, lamB, support_keep):
    Vmix = 0.5 * (V + V1)
    mask = row_support_mask(np.abs(V0) + np.abs(Vmix), support_keep)
    Vs = np.zeros_like(Vmix)
    Vs[mask, :] = Vmix[mask, :]
    Vhat = proj_stiefel(Vs)
    beta0hat, betahat = refit_beta(X, y, Vhat)
    Zhat = X @ Vhat
    Jhat = spcr_objective(X, y, Vhat, Zhat, beta0hat, betahat, w, lamV, lamB)
    Jreg = 0.5 / X.shape[0] * np.sum((y - beta0hat - X @ Vhat @ betahat) ** 2)
    return {"Vhat": Vhat, "Zhat": Zhat,
            "betahat": betahat, "beta0hat": beta0hat,
            "Jhat": Jhat, "Jreg": Jreg}


# ── ADMM solver ──────────────────────────────────────────────

def ADMM_SPCR(
    X, y, V_init,
    w=0.5, lamV=0.02, lamB=0.01,
    rho1=1.0, rho2=1.0, rho3=1.0,
    max_iter=300, tol_primal=7e-4, tol_dual=7e-4,
    min_iter=50, support_keep=24, use_woodbury=None
):
    n, p = X.shape
    k = V_init.shape[1]
    if use_woodbury is None:
        use_woodbury = p > 2 * n
    XtX_over_n = None if use_woodbury else (X.T @ X) / n

    V = proj_stiefel(V_init.copy())
    V0 = V.copy(); V1 = V.copy(); Z = X @ V
    beta0, beta = refit_beta(X, y, V)
    beta_clone = beta.copy()
    Gamma1 = np.zeros((p, k)); Gamma2 = np.zeros((p, k)); lam3 = np.zeros(k)
    V0_prev = V0.copy(); beta_clone_prev = beta_clone.copy()

    best_Jreg = np.inf; best_state = None
    trace_obj = []; trace_best = []
    trace_primal = []; trace_dual = []
    trace_sparsity = []

    for it in range(max_iter):
        RHS_V1 = (X.T @ np.outer(y - beta0, beta)) / n + rho2 * (V0 - Gamma2)
        if use_woodbury:
            V1 = solve_V1_sylvester_woodbury(X, beta, rho2, RHS_V1)
        else:
            V1 = solve_V1_sylvester(XtX_over_n, beta, rho2, RHS_V1)

        M = w * (X.T @ Z) / n + rho1 * (V0 - Gamma1)
        V = proj_stiefel(M)

        avg = (rho1 * (V + Gamma1) + rho2 * (V1 + Gamma2)) / (rho1 + rho2)
        V0 = soft(avg, lamV / (rho1 + rho2))
        Z = X @ V

        XV1 = X @ V1
        G = (XV1.T @ XV1) / n + rho3 * np.eye(k)
        h = (XV1.T @ (y - beta0)) / n + rho3 * (beta_clone - lam3)
        beta = solve(G, h)
        beta_clone = soft(beta + lam3, lamB / rho3)
        beta0 = float(np.mean(y - XV1 @ beta))

        r1 = V - V0; r2 = V1 - V0; r3 = beta - beta_clone
        Gamma1 += r1; Gamma2 += r2; lam3 += r3

        s1 = rho1 * (V0 - V0_prev)
        s2 = rho2 * (V0 - V0_prev)
        s3 = rho3 * (beta_clone - beta_clone_prev)
        primal_res = np.sqrt(np.sum(r1**2) + np.sum(r2**2) + np.sum(r3**2))
        dual_res   = np.sqrt(np.sum(s1**2) + np.sum(s2**2) + np.sum(s3**2))

        feasible = make_feasible_state(X, y, V, V0, V1, beta, beta_clone,
                                       w, lamV, lamB, support_keep)

        trace_obj.append(feasible["Jreg"])
        trace_best.append(min(best_Jreg, feasible["Jreg"]))
        trace_primal.append(primal_res)
        trace_dual.append(dual_res)
        trace_sparsity.append(np.mean(np.abs(feasible["Vhat"]) < 1e-8))

        # seçim: regression loss (Jreg) ile
        if feasible["Jreg"] < best_Jreg:
            best_Jreg = feasible["Jreg"]
            best_state = {
                "Vhat": feasible["Vhat"].copy(),
                "Zhat": feasible["Zhat"].copy(),
                "betahat": feasible["betahat"].copy(),
                "beta0hat": feasible["beta0hat"],
                "best_obj": feasible["Jhat"],
                "best_Jreg": feasible["Jreg"],
                "iter_best": it + 1
            }

        V0_prev = V0.copy()
        beta_clone_prev = beta_clone.copy()

        if (it + 1) >= min_iter and primal_res < tol_primal and dual_res < tol_dual:
            break

    out = best_state.copy()
    out["trace_obj"]      = np.array(trace_obj)
    out["trace_best"]     = np.array(trace_best)
    out["trace_primal"]   = np.array(trace_primal)
    out["trace_dual"]     = np.array(trace_dual)
    out["trace_sparsity"] = np.array(trace_sparsity)
    out["final_iter"]     = len(trace_obj)
    out["final_primal"]   = trace_primal[-1]
    out["final_dual"]     = trace_dual[-1]
    return out


# ── multi-start wrapper ──────────────────────────────────────

def multistart_ADMM(X, y, k, n_starts=50, rng=None, **admm_kwargs):
    """Hybrid multi-start:
       - candidate 0     : SVD init (same as single-start)
       - candidates 1..N/2 : PCA + threshold + perturbation
       - candidates N/2..N : dense random Stiefel
    """
    if rng is None:
        rng = np.random.default_rng()
    n, p = X.shape
    best_fit = None
    best_Jreg = np.inf
    all_Jregs = []

    # SVD yönleri (veri yapısından bilgi)
    _, _, Vt = svd(X, full_matrices=False)
    V_pca = Vt.T[:, :k]

    n_pca_thresh = n_starts // 2  # yarısı PCA-thresholded
    # candidate 0 = SVD, candidates 1..n_pca_thresh = PCA-thresh,
    # geri kalan = dense random

    for i in range(n_starts):
        if i == 0:
            # --- SVD init (single-start ile aynı) ---
            V0 = V_pca.copy()

        elif i <= n_pca_thresh:
            # --- PCA + threshold + perturbation ---
            V_thresh = V_pca.copy()
            # rastgele yüzdelik ile sıfırla (çeşitlilik sağlar)
            pct = rng.uniform(50, 85)
            cutoff = np.percentile(np.abs(V_thresh), pct)
            V_thresh[np.abs(V_thresh) < cutoff] = 0.0
            # perturbation
            noise_scale = rng.uniform(0.05, 0.3)
            V_thresh += noise_scale * rng.normal(size=V_thresh.shape)
            V0 = proj_stiefel(V_thresh)

        else:
            # --- dense random Stiefel ---
            V0 = proj_stiefel(rng.normal(size=(p, k)))

        fit = ADMM_SPCR(X, y, V0, **admm_kwargs)
        all_Jregs.append(fit["best_Jreg"])

        if fit["best_Jreg"] < best_Jreg:
            best_Jreg = fit["best_Jreg"]
            best_fit = fit

    best_fit["all_objs"] = np.array(all_Jregs)
    return best_fit


# ── convergence plots ────────────────────────────────────────

def plot_convergence(fit_single, fit_multi, title="", savepath=None):
    """4-panel convergence comparison: best obj, primal residual,
    sparsity, and final objective bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (1) best feasible objective over iterations
    ax = axes[0, 0]
    ax.plot(fit_single["trace_best"], label="Single-start", lw=1.8, color="#2c3e50")
    ax.plot(fit_multi["trace_best"],  label="Multi-start",  lw=1.8, color="#27ae60")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best feasible objective")
    ax.legend(); ax.set_title("Objective convergence")

    # (2) primal residual (log scale)
    ax = axes[0, 1]
    ax.semilogy(fit_single["trace_primal"], label="Single-start", lw=1.8, color="#2c3e50")
    ax.semilogy(fit_multi["trace_primal"],  label="Multi-start",  lw=1.8, color="#27ae60")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Primal residual")
    ax.legend(); ax.set_title("Primal residual")

    # (3) dual residual (log scale)
    ax = axes[1, 0]
    ax.semilogy(fit_single["trace_dual"], label="Single-start", lw=1.8, color="#2c3e50")
    ax.semilogy(fit_multi["trace_dual"],  label="Multi-start",  lw=1.8, color="#27ae60")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Dual residual")
    ax.legend(); ax.set_title("Dual residual")

    # (4) sparsity of Vhat over iterations
    ax = axes[1, 1]
    ax.plot(fit_single["trace_sparsity"], label="Single-start", lw=1.8, color="#2c3e50")
    ax.plot(fit_multi["trace_sparsity"],  label="Multi-start",  lw=1.8, color="#27ae60")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Sparsity of $\\hat{V}$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(); ax.set_title("Loading sparsity")

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_multistart_landscape(fit_multi, fit_single=None, title="", savepath=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    objs = fit_multi["all_objs"]  # Jreg değerleri
    ax.hist(objs, bins=min(30, len(objs)), color="#abebc6",
            edgecolor="black", alpha=0.85, label="Multi-start candidates")
    ax.axvline(fit_multi["best_Jreg"], color="#27ae60", lw=2.5,
               ls="--", label=f"Best multi-start ({fit_multi['best_Jreg']:.4f})")
    if fit_single is not None:
        ax.axvline(fit_single["best_Jreg"], color="#e74c3c", lw=2.5,
                   ls="-", label=f"Single-start ({fit_single['best_Jreg']:.4f})")
    ax.set_xlabel("Training regression loss $J_{reg}$")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title(title if title else "Regression loss landscape across initializations")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
