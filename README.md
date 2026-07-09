# Multi-start ADMM for SPCRsvd

Reference implementation and reproduction code for the paper

> **Stable Estimation of Sparse Principal Component Regression with SVD:
> A Multi-Start ADMM Framework for Low- and High-Dimensional Data**
> İsmail Yenilmez and Ersin Yılmaz.

The code estimates the SVD-based sparse principal component regression model
(SPCRsvd) with ADMM, and compares the standard single-start solver against a
multi-start strategy with hybrid initialization. It reproduces the simulation
study and the real-data applications reported in the paper.

## Method

SPCRsvd couples a regression loss for `y` with an SVD-type reconstruction loss
for `X`, under an `l1` penalty on the loadings `V` (constrained to the Stiefel
manifold) and on the regression coefficients `beta`. The objective is nonconvex,
so a local ADMM solver depends on its initialization. The multi-start procedure
generates several hybrid initial loadings (a data-driven SVD candidate,
thresholded and perturbed SVD candidates, and dense random Stiefel candidates),
refines each with the same ADMM updates, and keeps the candidate with the lowest
regression loss.

## Repository layout

```
spcrsvd-multistart-admm/
  src/
    spcrsvd_core.py          ADMM solver, multi-start wrapper, utilities, plots
    run_all_scenarios.py     simulation study (scenarios S1-S6)
    realdata_runner.py       real-data runner (resumable, time-budgeted)
    run_highdim_realdata.py  high-dimensional real-data script (gasoline NIR)
  demo/
    demo_quickstart.py       minimal single-vs-multi comparison
  requirements.txt
  LICENSE
  CITATION.cff
  DATA_AND_CODE_AVAILABILITY.md
```

## Installation

Python 3.9 or later.

```
pip install -r requirements.txt
```

## Quick start

A short demo that fits both estimators on a small synthetic high-dimensional
problem and prints the test RMSE of each:

```
cd demo
python demo_quickstart.py
```

## Reproducing the paper

### Simulation study (Section 5)

```
cd src
python run_all_scenarios.py --nrep 50 --nstarts 50
```

Outputs are written to `src/sim_results/`: `results_all.csv` (per-replicate
results), `summary_table.csv` and `summary_table.tex` (the Monte Carlo table),
and the boxplot, win-rate, convergence, and landscape figures.

A quick check on two scenarios:

```
python run_all_scenarios.py --nrep 3 --scenarios S1 S5
```

### Low-dimensional real data (Section 6.1)

The diabetes dataset ships with scikit-learn, so no download is needed:

```
cd src
python realdata_runner.py --dataset diabetes --nrep 50 --nstarts 50 --k 2
```

### High-dimensional real data (Section 6.2)

The gasoline NIR dataset (n = 60, p = 401, response = octane number) is public.
`run_highdim_realdata.py` downloads it once on first run, or you can place a
`gasoline.csv` in `src/realdata/` yourself:

```
cd src
python run_highdim_realdata.py --nrep 50 --nstarts 50 --k 2 --support_keep 60
```

Outputs are written to `src/realdata_results/`: the per-split CSV, a summary CSV,
a LaTeX table, and the boxplot, landscape, and convergence figures.

## Data

- Diabetes: `sklearn.datasets.load_diabetes` (Efron et al., 2004).
- Gasoline NIR: octane numbers and NIR spectra of 60 gasoline samples
  (Kalivas, 1997), obtained from the `pls` R package and mirrored publicly.

## Notes on settings

Tuning parameters are held fixed across replicates (`w = 0.5`, `lamV = 0.02`,
`lamB = 0.01`; `lamV = 0.015` for the low-dimensional scenario S5) so that the
comparison isolates the effect of the initialization strategy. The ADMM penalty
parameters are `rho1 = rho2 = rho3 = 2.0`. Both estimators use the same inner
ADMM updates and stopping rule.

## License

MIT License. See `LICENSE`.

## Citation

See `CITATION.cff`.
