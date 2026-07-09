"""
run_highdim_realdata.py
=======================
High-dimensional real-data application for the SPCRsvd paper:
single-start ADMM vs multi-start ADMM on the gasoline NIR dataset.

WHAT IT DOES
------------
1. Loads the gasoline NIR data (n=60, p=401 wavelengths, response = octane).
   - If realdata/gasoline.csv is present, it uses that.
   - Otherwise it tries to download it once from the octane-NIR GitHub repo.
2. Runs 50 random 70/30 train/test splits. In each split it standardizes X
   and centers y on the training part only, optionally screens the top
   --screen predictors by |correlation with y| on the training part, then
   fits both estimators with the same inner ADMM updates and stopping rule.
3. Writes, into realdata_results/:
      realdata_gasoline.csv          (per-split raw results)
      realdata_gasoline_summary.csv  (means, win rates, Wilcoxon p)
      realdata_gasoline_table.tex    (LaTeX table)
      realdata_gasoline_boxplot.png  (RMSE + objective boxplots)
      realdata_gasoline_landscape.png(J_reg spread across starts, one split)
      realdata_gasoline_convergence.png (single vs multi convergence, one split)
4. Prints a summary block to the console.

Run: python run_highdim_realdata.py
Defaults: --nrep 50 --nstarts 50 --k 2 --screen 401 --support_keep 60 --seed 7000
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from numpy.linalg import svd
from scipy.stats import wilcoxon

from spcrsvd_core import (
    ADMM_SPCR, multistart_ADMM,
    standardize_from_train, center_y_from_train, rmse,
    plot_convergence, plot_multistart_landscape,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(HERE, "realdata")
OUTDIR = os.path.join(HERE, "realdata_results")
os.makedirs(DATADIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

GASOLINE_URL = ("https://raw.githubusercontent.com/gustavovelascoh/"
                "octane-NIR/master/gasoline.csv")


def load_gasoline():
    """Return X (n x 401), y (n,), after obtaining gasoline.csv if needed."""
    local = os.path.join(DATADIR, "gasoline.csv")
    if not os.path.exists(local):
        print("gasoline.csv not found locally; downloading once ...")
        import urllib.request
        urllib.request.urlretrieve(GASOLINE_URL, local)
        print("  saved to", local)
    # The CSV has an unnamed index column, then 'octane', then NIR.xxx columns.
    df = pd.read_csv(local, index_col=0)
    if "octane" not in df.columns:
        y = df.iloc[:, 0].values.astype(float)
        X = df.iloc[:, 1:].values.astype(float)
    else:
        y = df["octane"].values.astype(float)
        X = df.drop(columns=["octane"]).values.astype(float)
    print("gasoline loaded: X =", X.shape, " y in [%.1f, %.1f]" % (y.min(), y.max()))
    return X, y


def screen_by_correlation(Xtr, ytr, Xte, n_keep):
    if n_keep is None or n_keep >= Xtr.shape[1]:
        return Xtr, Xte
    yc = ytr - ytr.mean()
    Xc = Xtr - Xtr.mean(axis=0, keepdims=True)
    denom = np.sqrt((Xc ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-12
    corr = np.abs((Xc * yc[:, None]).sum(0) / denom)
    idx = np.argsort(-corr)[:n_keep]
    return Xtr[:, idx], Xte[:, idx]


ADMM_DEFAULTS = dict(
    w=0.5, lamV=0.02, lamB=0.01,
    rho1=2.0, rho2=2.0, rho3=2.0,
    max_iter=300, tol_primal=7e-4, tol_dual=7e-4, min_iter=50,
)


def run_one_split(X, y, seed, k, support_keep, n_starts, screen, keep_fits=False):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    ntr = int(0.7 * n)
    Xtr_raw, Xte_raw = X[idx[:ntr]], X[idx[ntr:]]
    ytr_raw, yte_raw = y[idx[:ntr]], y[idx[ntr:]]
    Xtr_raw, Xte_raw = screen_by_correlation(Xtr_raw, ytr_raw, Xte_raw, screen)
    Xtr, Xte = standardize_from_train(Xtr_raw, Xte_raw)
    ytr, yte = center_y_from_train(ytr_raw, yte_raw)

    admm_kw = dict(ADMM_DEFAULTS)
    admm_kw["support_keep"] = support_keep

    _, _, Vt = svd(Xtr, full_matrices=False)
    V0_svd = Vt.T[:, :k]
    t0 = time.time()
    fs = ADMM_SPCR(Xtr, ytr, V0_svd, **admm_kw)
    ts = time.time() - t0
    yhat_s = fs["beta0hat"] + Xte @ fs["Vhat"] @ fs["betahat"]
    rmse_s = rmse(yte, yhat_s)
    spar_s = float(np.mean(np.abs(fs["Vhat"]) < 1e-8))

    t0 = time.time()
    fm = multistart_ADMM(Xtr, ytr, k=k, n_starts=n_starts, rng=rng, **admm_kw)
    tm = time.time() - t0
    yhat_m = fm["beta0hat"] + Xte @ fm["Vhat"] @ fm["betahat"]
    rmse_m = rmse(yte, yhat_m)
    spar_m = float(np.mean(np.abs(fm["Vhat"]) < 1e-8))

    row = {
        "seed": seed,
        "rmse_single": rmse_s, "rmse_multi": rmse_m,
        "obj_single": fs["best_obj"], "obj_multi": fm["best_obj"],
        "Jreg_single": fs["best_Jreg"], "Jreg_multi": fm["best_Jreg"],
        "sparsity_single": spar_s, "sparsity_multi": spar_m,
        "time_single": ts, "time_multi": tm,
        "win_rmse": int(rmse_m < rmse_s),
        "win_obj": int(fm["best_Jreg"] < fs["best_Jreg"]),
        "rmse_improvement": float((rmse_s - rmse_m) / rmse_s * 100),
    }
    if keep_fits:
        return row, fs, fm
    return row


def make_boxplot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (m, lab) in zip(axes, [("rmse", "Test RMSE"),
                                   ("obj", "Training objective $J$")]):
        bp = ax.boxplot([df[f"{m}_single"], df[f"{m}_multi"]],
                        tick_labels=["Single", "Multi"], widths=0.5,
                        patch_artist=True)
        bp["boxes"][0].set_facecolor("#d4e6f1")
        bp["boxes"][1].set_facecolor("#abebc6")
        ax.set_ylabel(lab)
    fig.suptitle("Gasoline NIR: single-start vs multi-start", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("  figure ->", path)


def save_latex_table(s, path):
    with open(path, "w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{High-dimensional real-data results on the gasoline "
                "NIR dataset ($n=60$, $p=401$, $k=2$), averaged over $50$ random "
                "$70/30$ train--test splits. RMSE is on the octane scale. "
                "$J_{\\mathrm{reg}}$ is the training regression loss; "
                "``Sparsity'' is the fraction of near-zero loading entries. "
                "The last column is the one-sided paired Wilcoxon $p$-value for "
                "the hypothesis that multi-start reduces RMSE.}\n")
        f.write("\\label{tab:realdata_hd}\n\\smallskip\n")
        f.write("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n")
        f.write("Method & Test RMSE & $J_{\\mathrm{reg}}$ & Sparsity & "
                "RMSE win (\\%) & Wilcoxon $p$ \\\\\n\\midrule\n")
        f.write("Single-start & $%.3f \\pm %.3f$ & $%.3f$ & $%.2f$ & -- & "
                "\\multirow{2}{*}{$%.3g$} \\\\\n" % (
                    s["rmse_single_mean"], s["rmse_single_std"],
                    s["Jreg_single_mean"], s["sparsity_single_mean"],
                    s["wilcoxon_rmse_p"]))
        f.write("Multi-start & $%.3f \\pm %.3f$ & $%.3f$ & $%.2f$ & $%.0f$ & \\\\\n" % (
            s["rmse_multi_mean"], s["rmse_multi_std"],
            s["Jreg_multi_mean"], s["sparsity_multi_mean"],
            100 * s["win_rmse"]))
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print("  table  ->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrep", type=int, default=50)
    ap.add_argument("--nstarts", type=int, default=50)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--screen", type=int, default=401)
    ap.add_argument("--support_keep", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7000)
    args = ap.parse_args()

    X, y = load_gasoline()
    screen = None if args.screen >= X.shape[1] else args.screen
    print("Protocol: nrep=%d nstarts=%d k=%d screen=%s support_keep=%d"
          % (args.nrep, args.nstarts, args.k, screen, args.support_keep))

    rows, first_fits = [], None
    t0 = time.time()
    for rep in range(args.nrep):
        seed = args.seed + 1000 * rep
        if rep == 0:
            r, fs, fm = run_one_split(X, y, seed, args.k, args.support_keep,
                                      args.nstarts, screen, keep_fits=True)
            first_fits = (fs, fm)
        else:
            r = run_one_split(X, y, seed, args.k, args.support_keep,
                              args.nstarts, screen)
        rows.append(r)
        print("  split %2d/%d  RMSE s=%.4f m=%.4f  Jreg s=%.4f m=%.4f"
              % (rep + 1, args.nrep, r["rmse_single"], r["rmse_multi"],
                 r["Jreg_single"], r["Jreg_multi"]))
    print("Total %.1fs" % (time.time() - t0))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "realdata_gasoline.csv"), index=False)

    w_rmse = wilcoxon(df.rmse_single, df.rmse_multi, alternative="greater").pvalue
    s = {
        "dataset": "Gasoline NIR (n=%d, p=%d)" % (X.shape[0], X.shape[1]),
        "rmse_single_mean": df.rmse_single.mean(), "rmse_single_std": df.rmse_single.std(),
        "rmse_multi_mean": df.rmse_multi.mean(), "rmse_multi_std": df.rmse_multi.std(),
        "obj_single_mean": df.obj_single.mean(), "obj_multi_mean": df.obj_multi.mean(),
        "Jreg_single_mean": df.Jreg_single.mean(), "Jreg_multi_mean": df.Jreg_multi.mean(),
        "sparsity_single_mean": df.sparsity_single.mean(),
        "sparsity_multi_mean": df.sparsity_multi.mean(),
        "win_rmse": df.win_rmse.mean(), "win_obj": df.win_obj.mean(),
        "rmse_improvement_mean": df.rmse_improvement.mean(),
        "wilcoxon_rmse_p": w_rmse,
        "time_single_mean": df.time_single.mean(), "time_multi_mean": df.time_multi.mean(),
    }
    pd.DataFrame([s]).to_csv(
        os.path.join(OUTDIR, "realdata_gasoline_summary.csv"), index=False)
    save_latex_table(s, os.path.join(OUTDIR, "realdata_gasoline_table.tex"))
    make_boxplot(df, os.path.join(OUTDIR, "realdata_gasoline_boxplot.png"))
    if first_fits is not None:
        fs, fm = first_fits
        plot_convergence(fs, fm, title="Gasoline NIR - convergence (split 1)",
                         savepath=os.path.join(OUTDIR, "realdata_gasoline_convergence.png"))
        plot_multistart_landscape(
            fm, fs, title="Gasoline NIR - J_reg across initializations (split 1)",
            savepath=os.path.join(OUTDIR, "realdata_gasoline_landscape.png"))
        print("  figures -> convergence, landscape")

    print("\n=== SUMMARY (paste this back) ===")
    for kk in ["dataset", "rmse_single_mean", "rmse_single_std",
               "rmse_multi_mean", "rmse_multi_std", "win_rmse",
               "rmse_improvement_mean", "wilcoxon_rmse_p",
               "Jreg_single_mean", "Jreg_multi_mean", "win_obj",
               "sparsity_single_mean", "sparsity_multi_mean",
               "obj_single_mean", "obj_multi_mean",
               "time_single_mean", "time_multi_mean"]:
        print("  %s: %s" % (kk, s[kk]))


if __name__ == "__main__":
    main()
