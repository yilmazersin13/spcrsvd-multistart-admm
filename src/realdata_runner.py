
import argparse
import os
import time
import numpy as np
import pandas as pd
from numpy.linalg import svd
from scipy.stats import wilcoxon
from concurrent.futures import ProcessPoolExecutor

from spcrsvd_core import (
    ADMM_SPCR, multistart_ADMM,
    standardize_from_train, center_y_from_train, rmse,
)

OUTDIR = "realdata_results"
os.makedirs(OUTDIR, exist_ok=True)

ADMM_DEFAULTS = dict(
    w=0.5, lamV=0.02, lamB=0.01,
    rho1=2.0, rho2=2.0, rho3=2.0,
    max_iter=300, tol_primal=7e-4, tol_dual=7e-4, min_iter=50,
)


def load_dataset(name, datafile=None):
    if name == "diabetes":
        from sklearn.datasets import load_diabetes
        d = load_diabetes()
        return d.data.astype(float), d.target.astype(float), "Diabetes (n=442, p=10)"
    if name == "gasoline":
        z = np.load(datafile)
        return z["X"].astype(float), z["y"].astype(float), \
            "Gasoline NIR (n=%d, p=%d)" % (z["X"].shape[0], z["X"].shape[1])
    if name == "csv":
        df = pd.read_csv(datafile)
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values.astype(float)
        return X, y, "%s (n=%d, p=%d)" % (os.path.basename(datafile), X.shape[0], X.shape[1])
    raise ValueError("unknown dataset %s" % name)


def screen_by_correlation(Xtr, ytr, Xte, n_keep):
    if n_keep is None or n_keep >= Xtr.shape[1]:
        return Xtr, Xte, np.arange(Xtr.shape[1])
    yc = ytr - ytr.mean()
    Xc = Xtr - Xtr.mean(axis=0, keepdims=True)
    denom = (np.sqrt((Xc ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-12)
    corr = np.abs((Xc * yc[:, None]).sum(0) / denom)
    idx = np.argsort(-corr)[:n_keep]
    return Xtr[:, idx], Xte[:, idx], idx


def run_one_split(X, y, seed, k, support_keep, n_starts, screen):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    ntr = int(0.7 * n)
    Xtr_raw, Xte_raw = X[idx[:ntr]], X[idx[ntr:]]
    ytr_raw, yte_raw = y[idx[:ntr]], y[idx[ntr:]]
    Xtr_raw, Xte_raw, _ = screen_by_correlation(Xtr_raw, ytr_raw, Xte_raw, screen)
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

    return {
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


def _worker(pack):
    return run_one_split(*pack)


def write_summary(csv_path, desc, tag, nrep):
    df = pd.read_csv(csv_path)
    w_rmse = wilcoxon(df.rmse_single, df.rmse_multi, alternative="greater").pvalue
    summary = {
        "dataset": desc,
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
    pd.DataFrame([summary]).to_csv(
        os.path.join(OUTDIR, "realdata_%s_summary.csv" % tag), index=False)
    print("\nSUMMARY")
    for kk, vv in summary.items():
        print("  %s: %s" % (kk, vv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--datafile", default=None)
    ap.add_argument("--nrep", type=int, default=50)
    ap.add_argument("--nstarts", type=int, default=50)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--support_keep", type=int, default=None)
    ap.add_argument("--screen", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--time_budget", type=float, default=1e9)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    X, y, desc = load_dataset(args.dataset, args.datafile)
    p_eff = X.shape[1] if args.screen is None else min(args.screen, X.shape[1])
    support_keep = args.support_keep if args.support_keep is not None else p_eff
    tag = args.dataset
    csv_path = os.path.join(OUTDIR, "realdata_%s.csv" % tag)

    all_seeds = [args.seed + 1000 * rep for rep in range(args.nrep)]
    done_seeds = set()
    if os.path.exists(csv_path):
        done_seeds = set(pd.read_csv(csv_path)["seed"].tolist())
    todo = [s for s in all_seeds if s not in done_seeds]
    print("Dataset : %s" % desc)
    print("k=%d support_keep=%s screen=%s nrep=%d nstarts=%d done=%d todo=%d"
          % (args.k, support_keep, args.screen, args.nrep, args.nstarts,
             len(done_seeds), len(todo)))

    t0 = time.time()
    packs = [(X, y, s, args.k, support_keep, args.nstarts, args.screen) for s in todo]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i in range(0, len(packs), args.workers):
            if time.time() - t0 > args.time_budget:
                break
            batch = packs[i:i + args.workers]
            for r in ex.map(_worker, batch):
                header = not os.path.exists(csv_path)
                pd.DataFrame([r]).to_csv(csv_path, mode="a", header=header, index=False)
                print("  seed %d RMSE s=%.4f m=%.4f Jreg s=%.4f m=%.4f (%.1fs)"
                      % (r["seed"], r["rmse_single"], r["rmse_multi"],
                         r["Jreg_single"], r["Jreg_multi"], time.time() - t0))

    ndone = len(pd.read_csv(csv_path))
    print("Chunk done in %.1fs. Stored %d/%d splits."
          % (time.time() - t0, ndone, args.nrep))
    if ndone < args.nrep:
        print("INCOMPLETE - rerun the same command to continue.")
        return
    write_summary(csv_path, desc, tag, args.nrep)


if __name__ == "__main__":
    main()
