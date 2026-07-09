"""
run_all_scenarios.py
====================
SPCRsvd simulation: single-start ADMM vs multi-start ADMM.

Usage:
    python run_all_scenarios.py                        # 5 reps, all scenarios
    python run_all_scenarios.py --nrep 50              # full run
    python run_all_scenarios.py --nrep 3 --scenarios S1 S5   # quick test
"""

import argparse
import time
import os
import numpy as np
from numpy.linalg import svd
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spcrsvd_core import (
    generate_dgp, standardize_from_train, center_y_from_train,
    ADMM_SPCR, multistart_ADMM, sinThetaF, rmse,
    plot_convergence, plot_multistart_landscape
)

# ── output directory ────────────────────────────────────────
OUTDIR = "sim_results"
os.makedirs(OUTDIR, exist_ok=True)

# ── shared ADMM settings ────────────────────────────────────
ADMM_DEFAULTS = dict(
    w=0.5,
    lamV=0.02,
    lamB=0.01,
    rho1=2.0,
    rho2=2.0,
    rho3=2.0,
    max_iter=300,
    tol_primal=7e-4,
    tol_dual=7e-4,
    min_iter=50,
)

N_STARTS = 50

# ── scenario definitions ────────────────────────────────────
# Each scenario specifies DGP params + optional ADMM overrides.
# ADMM params not listed here fall back to ADMM_DEFAULTS.

SCENARIOS = {
    "S1": dict(
        label="S1 (baseline)",
        n=120, p=500, k=2,
        nuisance_rank=4, nuisance_strength=15.0, signal_strength=1.5,
        support_size=12, noise_y=0.45,
        rho_noise=0.30, noise_x=0.35,
        # ADMM overrides (if any)
    ),
    "S2": dict(
        label="S2 (strong nuisance)",
        n=120, p=500, k=2,
        nuisance_rank=6, nuisance_strength=30.0, signal_strength=1.5,
        support_size=10, noise_y=0.45,
        rho_noise=0.30, noise_x=0.35,
    ),
    "S3": dict(
        label="S3 (ultra-high dim)",
        n=100, p=1000, k=2,
        nuisance_rank=6, nuisance_strength=21.0, signal_strength=1.5,
        support_size=8, noise_y=0.45,
        rho_noise=0.30, noise_x=0.35,
    ),
    "S4": dict(
        label="S4 (more components)",
        n=150, p=500, k=4,
        nuisance_rank=4, nuisance_strength=15.0, signal_strength=1.5,
        support_size=10, noise_y=0.50,
        rho_noise=0.30, noise_x=0.35,
    ),
    "S5": dict(
        label="S5 (low-dim)",
        n=200, p=50, k=2,
        nuisance_rank=3, nuisance_strength=15.0, signal_strength=1.5,
        support_size=8, noise_y=0.45,
        rho_noise=0.30, noise_x=0.35,
        # ADMM override: lighter penalty for low-dim
        lamV=0.015,
    ),
    "S6": dict(
        label="S6 (low SNR)",
        n=120, p=500, k=2,
        nuisance_rank=4, nuisance_strength=15.0, signal_strength=1.5,
        support_size=12, noise_y=1.20,
        rho_noise=0.30, noise_x=0.35,
    ),
}

# DGP parameter names (to separate from ADMM overrides)
DGP_KEYS = {"n", "p", "k", "support_size", "nuisance_rank",
            "nuisance_strength", "signal_strength",
            "rho_noise", "noise_x", "noise_y", "label"}


def get_admm_kwargs(scenario):
    """Merge scenario-level ADMM overrides with defaults."""
    kw = dict(ADMM_DEFAULTS)
    for key, val in scenario.items():
        if key not in DGP_KEYS and key in ADMM_DEFAULTS:
            kw[key] = val
    return kw


# ── single replicate ────────────────────────────────────────

def run_one_replicate(scenario_key, scenario, seed, n_starts=50, store_fits=False):
    sc = scenario
    rng = np.random.default_rng(seed)

    # generate data
    X, y, B_true = generate_dgp(
        n=sc["n"], p=sc["p"], k=sc["k"],
        support_size=sc["support_size"],
        nuisance_rank=sc["nuisance_rank"],
        nuisance_strength=sc["nuisance_strength"],
        signal_strength=sc["signal_strength"],
        rho_noise=sc["rho_noise"],
        noise_x=sc["noise_x"],
        noise_y=sc["noise_y"],
        rng=rng,
    )

    # train/test split
    idx = rng.permutation(sc["n"])
    ntr = int(0.7 * sc["n"])
    Xtr_raw, Xte_raw = X[idx[:ntr]], X[idx[ntr:]]
    ytr_raw, yte_raw = y[idx[:ntr]], y[idx[ntr:]]
    Xtr, Xte = standardize_from_train(Xtr_raw, Xte_raw)
    ytr, yte = center_y_from_train(ytr_raw, yte_raw)

    k = sc["k"]
    support_keep = k * sc["support_size"]
    admm_kw = get_admm_kwargs(sc)
    admm_kw["support_keep"] = support_keep

    # ── single-start (SVD init) ──
    _, _, Vt = svd(Xtr, full_matrices=False)
    V0_svd = Vt.T[:, :k]

    t0 = time.time()
    fit_single = ADMM_SPCR(Xtr, ytr, V0_svd, **admm_kw)
    time_single = time.time() - t0

    yhat_s = fit_single["beta0hat"] + Xte @ fit_single["Vhat"] @ fit_single["betahat"]
    rmse_s  = rmse(yte, yhat_s)
    theta_s = sinThetaF(fit_single["Vhat"], B_true)
    spar_s  = np.mean(np.abs(fit_single["Vhat"]) < 1e-8)

    # ── multi-start (hybrid) ──
    t0 = time.time()
    fit_multi = multistart_ADMM(
        Xtr, ytr, k=k, n_starts=n_starts,
        rng=rng, **admm_kw
    )
    time_multi = time.time() - t0

    yhat_m = fit_multi["beta0hat"] + Xte @ fit_multi["Vhat"] @ fit_multi["betahat"]
    rmse_m  = rmse(yte, yhat_m)
    theta_m = sinThetaF(fit_multi["Vhat"], B_true)
    spar_m  = np.mean(np.abs(fit_multi["Vhat"]) < 1e-8)

    # objective spread + failure rate
    all_Jregs = fit_multi["all_objs"]
    obj_spread_std   = float(np.std(all_Jregs))
    obj_spread_range = float(np.ptp(all_Jregs))

    # failure rate: yüzde kaç candidate single-start'tan kötü
    single_Jreg = fit_single["best_Jreg"]
    n_worse = np.sum(all_Jregs > single_Jreg)
    failure_rate = float(n_worse / len(all_Jregs))

    # RMSE improvement yüzdesi
    rmse_improvement = float((rmse_s - rmse_m) / rmse_s * 100)

    result = {
        "scenario": scenario_key, "seed": seed,
        # single
        "rmse_single":     rmse_s,
        "obj_single":      fit_single["best_obj"],
        "Jreg_single":     fit_single["best_Jreg"],
        "sintheta_single": theta_s,
        "sparsity_single": spar_s,
        "iters_single":    fit_single["final_iter"],
        "time_single":     time_single,
        # multi
        "rmse_multi":      rmse_m,
        "obj_multi":       fit_multi["best_obj"],
        "Jreg_multi":      fit_multi["best_Jreg"],
        "sintheta_multi":  theta_m,
        "sparsity_multi":  spar_m,
        "iters_multi":     fit_multi["final_iter"],
        "time_multi":      time_multi,
        # diagnostics
        "obj_spread_std":    obj_spread_std,
        "obj_spread_range":  obj_spread_range,
        "failure_rate":      failure_rate,
        "rmse_improvement":  rmse_improvement,
        # wins
        "win_rmse":  int(rmse_m < rmse_s),
        "win_obj":   int(fit_multi["best_Jreg"] < fit_single["best_Jreg"]),
        "win_theta": int(theta_m < theta_s),
    }

    if store_fits:
        result["_fit_single"] = fit_single
        result["_fit_multi"]  = fit_multi

    return result

# ── MC loop per scenario ────────────────────────────────────

def run_scenario(scenario_key, scenario, n_rep, base_seed=5000, n_starts=50):
    print(f"\n{'='*70}")
    print(f"  {scenario['label']}   (n={scenario['n']}, p={scenario['p']}, "
          f"k={scenario['k']}, r={scenario['nuisance_rank']})")
    print(f"{'='*70}")

    rows = []
    first_fits = None  # store fits from first replicate for convergence plots

    for rep in range(n_rep):
        seed = base_seed + 1000 * rep
        t0 = time.time()
        store = (rep == 0)  # only store full fits for first replicate
        row = run_one_replicate(scenario_key, scenario, seed,
                                n_starts=n_starts, store_fits=store)
        elapsed = time.time() - t0

        # extract fits before removing them from the row
        if store and "_fit_single" in row:
            first_fits = {
                "single": row.pop("_fit_single"),
                "multi":  row.pop("_fit_multi"),
            }

        rows.append(row)
        print(f"  rep {rep+1:3d}/{n_rep}  "
              f"RMSE s={row['rmse_single']:.4f} m={row['rmse_multi']:.4f}  "
              f"Jreg s={row['Jreg_single']:.4f} m={row['Jreg_multi']:.4f}  "
              f"sinT s={row['sintheta_single']:.4f} m={row['sintheta_multi']:.4f}  "
              f"({elapsed:.1f}s)")

    # save convergence plots for first replicate
    if first_fits is not None:
        plot_convergence(
            first_fits["single"], first_fits["multi"],
            title=f"{scenario['label']} — convergence (rep 1)",
            savepath=os.path.join(OUTDIR, f"convergence_{scenario_key}.png")
        )
        print(f"  Convergence plot → {OUTDIR}/convergence_{scenario_key}.png")

        plot_multistart_landscape(
            first_fits["multi"], first_fits["single"],
            title=f"{scenario['label']} — objective landscape (rep 1)",
            savepath=os.path.join(OUTDIR, f"landscape_{scenario_key}.png")
        )
        print(f"  Landscape plot   → {OUTDIR}/landscape_{scenario_key}.png")

    return rows


# ── summary ──────────────────────────────────────────────────

def make_summary(df):
    metrics = ["rmse", "obj", "sintheta", "sparsity"]
    rows = []
    for sc in df["scenario"].unique():
        sub = df[df["scenario"] == sc]
        row = {"scenario": sc, "n_rep": len(sub)}
        for m in metrics:
            for method in ["single", "multi"]:
                col = f"{m}_{method}"
                row[f"{m}_{method}_mean"] = sub[col].mean()
                row[f"{m}_{method}_std"]  = sub[col].std()
        row["win_rmse"]  = sub["win_rmse"].mean()
        row["win_obj"]   = sub["win_obj"].mean()
        row["win_theta"] = sub["win_theta"].mean()
        row["obj_spread_std_mean"]  = sub["obj_spread_std"].mean()
        row["failure_rate_mean"]    = sub["failure_rate"].mean()
        row["rmse_improvement_mean"] = sub["rmse_improvement"].mean()
        row["time_single_mean"] = sub["time_single"].mean()
        row["time_multi_mean"]  = sub["time_multi"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(summary):
    print(f"\n{'='*120}")
    print("  SUMMARY TABLE  (s = single-start,  m = multi-start)")
    print(f"{'='*120}")
    hdr = (f"{'':6s}  {'RMSE (s / m)':>28s}  {'sinΘ (s / m)':>28s}  "
           f"{'win%':>5s}  {'improv%':>7s}  {'fail%':>6s}")
    print(hdr)
    print("-" * 120)
    for _, r in summary.iterrows():
        def fmtpm(m, s):
            return f"{m:.3f}±{s:.3f}"
        rmse_str  = (f"{fmtpm(r['rmse_single_mean'],  r['rmse_single_std'])} / "
                     f"{fmtpm(r['rmse_multi_mean'],   r['rmse_multi_std'])}")
        theta_str = (f"{fmtpm(r['sintheta_single_mean'], r['sintheta_single_std'])} / "
                     f"{fmtpm(r['sintheta_multi_mean'],  r['sintheta_multi_std'])}")
        print(f"{r['scenario']:6s}  {rmse_str}  {theta_str}  "
              f"{100*r['win_rmse']:4.0f}%  "
              f"{r.get('rmse_improvement_mean', 0):+6.1f}%  "
              f"{100*r.get('failure_rate_mean', 0):5.1f}%")


# ── LaTeX table ──────────────────────────────────────────────

def save_latex_table(summary, path):
    with open(path, "w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Monte Carlo results: mean $\\pm$ std across replicates. "
                "Win rate is the proportion of replicates where multi-start "
                "achieves strictly lower test RMSE.}\n")
        f.write("\\label{tab:mc_results}\n\\smallskip\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{@{}l cc cc cc c@{}}\n\\toprule\n")
        f.write(" & \\multicolumn{2}{c}{Test RMSE} "
                "& \\multicolumn{2}{c}{Objective $J$} "
                "& \\multicolumn{2}{c}{$\\|\\sin\\Theta\\|_F$} "
                "& Win \\\\\n")
        f.write("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} "
                "\\cmidrule(lr){6-7} \\cmidrule(lr){8-8}\n")
        f.write("Scenario & Single & Multi & Single & Multi "
                "& Single & Multi & (\\%) \\\\\n")
        f.write("\\midrule\n")

        for _, r in summary.iterrows():
            def pm(m, s):
                return f"${m:.3f}" + " \\pm " + f"{s:.3f}$"
            f.write(
                f"{r['scenario']} & "
                f"{pm(r['rmse_single_mean'],    r['rmse_single_std'])} & "
                f"{pm(r['rmse_multi_mean'],     r['rmse_multi_std'])} & "
                f"{pm(r['obj_single_mean'],     r['obj_single_std'])} & "
                f"{pm(r['obj_multi_mean'],      r['obj_multi_std'])} & "
                f"{pm(r['sintheta_single_mean'],r['sintheta_single_std'])} & "
                f"{pm(r['sintheta_multi_mean'], r['sintheta_multi_std'])} & "
                f"{100*r['win_rmse']:.0f} \\\\\n"
            )

        f.write("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    print(f"  LaTeX table → {path}")


# ── plots ────────────────────────────────────────────────────

def make_plots(df):
    scenarios = list(df["scenario"].unique())
    n_sc = len(scenarios)

    for metric, ylabel in [("rmse",     "Test RMSE"),
                            ("obj",      "Best objective $J$"),
                            ("sintheta", "Subspace error $\\|\\sin\\Theta\\|_F$")]:
        fig, axes = plt.subplots(1, n_sc, figsize=(3.2 * n_sc, 4), sharey=False)
        if n_sc == 1:
            axes = [axes]
        for ax, sc in zip(axes, scenarios):
            sub = df[df["scenario"] == sc]
            data = [sub[f"{metric}_single"].values,
                    sub[f"{metric}_multi"].values]
            bp = ax.boxplot(data, tick_labels=["Single", "Multi"],
                            widths=0.5, patch_artist=True)
            bp["boxes"][0].set_facecolor("#d4e6f1")
            bp["boxes"][1].set_facecolor("#abebc6")
            ax.set_title(sc, fontsize=10)
            if ax == axes[0]:
                ax.set_ylabel(ylabel)
        fig.suptitle(ylabel, fontsize=12, y=1.02)
        fig.tight_layout()
        path = os.path.join(OUTDIR, f"boxplot_{metric}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot → {path}")

    # win-rate bar chart
    summary = make_summary(df)
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * n_sc), 4))
    x = np.arange(n_sc)
    bw = 0.25
    ax.bar(x - bw, summary["win_rmse"]  * 100, bw,
           label="RMSE",     color="#d4e6f1", edgecolor="black")
    ax.bar(x,      summary["win_obj"]   * 100, bw,
           label="Obj",      color="#abebc6", edgecolor="black")
    ax.bar(x + bw, summary["win_theta"] * 100, bw,
           label="$\\sin\\Theta$", color="#f9e79f", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["scenario"])
    ax.set_ylabel("Multi-start win rate (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.axhline(50, color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "winrate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot → {path}")

    # objective spread
    fig, axes = plt.subplots(1, n_sc, figsize=(3.2 * n_sc, 4), sharey=False)
    if n_sc == 1:
        axes = [axes]
    for ax, sc in zip(axes, scenarios):
        sub = df[df["scenario"] == sc]
        ax.boxplot(sub["obj_spread_std"].values, widths=0.5,
                   patch_artist=True,
                   boxprops=dict(facecolor="#fadbd8"))
        ax.set_title(sc, fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Obj spread (std across starts)")
        ax.set_xticklabels([""])
    fig.suptitle("Initialization sensitivity: objective spread", fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "obj_spread.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {path}")


# ── main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SPCRsvd simulation: single vs multi-start ADMM")
    parser.add_argument("--nrep", type=int, default=5,
                        help="MC replicates per scenario (default: 5)")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Run only these, e.g. --scenarios S1 S5")
    parser.add_argument("--seed", type=int, default=5000,
                        help="Base seed (default: 5000)")
    parser.add_argument("--nstarts", type=int, default=None,
                        help=f"Multi-start pool size (default: {N_STARTS})")
    args = parser.parse_args()

    n_starts = args.nstarts if args.nstarts is not None else N_STARTS

    sc_keys = args.scenarios if args.scenarios else list(SCENARIOS.keys())
    print(f"Scenarios : {sc_keys}")
    print(f"Replicates: {args.nrep}")
    print(f"N_starts  : {n_starts}")
    print(f"ADMM defaults: w={ADMM_DEFAULTS['w']}, "
          f"lamV={ADMM_DEFAULTS['lamV']}, lamB={ADMM_DEFAULTS['lamB']}")

    all_rows = []
    t_total = time.time()

    for sc_key in sc_keys:
        rows = run_scenario(sc_key, SCENARIOS[sc_key],
                            n_rep=args.nrep, base_seed=args.seed,
                            n_starts=n_starts)
        all_rows.extend(rows)

    elapsed = time.time() - t_total
    print(f"\nTotal time: {elapsed/60:.1f} min")

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTDIR, "results_all.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Full results → {csv_path}")

    summary = make_summary(df)
    summary.to_csv(os.path.join(OUTDIR, "summary_table.csv"), index=False)
    print_summary(summary)
    save_latex_table(summary, os.path.join(OUTDIR, "summary_table.tex"))
    make_plots(df)
    print(f"\nAll outputs in ./{OUTDIR}/")


if __name__ == "__main__":
    main()
