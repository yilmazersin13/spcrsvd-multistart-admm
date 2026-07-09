# Multi-start ADMM for SPCRsvd


**Stable Estimation of Sparse Principal Component Regression with SVD:  
A Multi-Start ADMM Framework for Low- and High-Dimensional Data**  
\.{I}smail Yenilmez and Ersin Y{\i}lmaz

The code fits the SVD-based sparse principal component regression model
(SPCRsvd) by ADMM. It compares the usual single-start ADMM estimator with a
multi-start ADMM estimator. The main purpose is to see whether using several
initial values makes SPCRsvd estimation more reliable for prediction.

## Main Idea

SPCRsvd uses one objective function for prediction, reconstruction, and
sparsity. This makes the problem useful, but also difficult, because the
objective is nonconvex. Therefore, the ADMM result may depend on the starting
values.

The multi-start method uses several initial loading matrices. These include an
SVD-based start, perturbed SVD-based starts, and random Stiefel starts. Each
start is refined by the same ADMM updates. At the end, the method keeps the
candidate with the lowest regression loss.

## Files

```text
Multi_ADMM Paper/
  main.tex                         original paper file
  main_rewritten.tex               rewritten paper file
  references.bib                   references
  Multi_ADMM_Paper_Ismail_Yenilmez (1).pdf

  Code/
    spcrsvd_core.py                main ADMM and multi-start functions
    run_all_scenarios_v1.py        simulation study, scenarios S1-S6
    realdata_runner.py             real-data runner
    run_highdim_realdata.py        gasoline NIR data application
    run_script_spcr.py             small runner script
    realdata/                      real-data files
    sim_results/                   simulation outputs
    realdata_results/              real-data outputs
```

## Requirements

The scripts use Python and the common scientific packages:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
```

Install them if they are not already available:

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

## Run the Simulation Study

From the `Code` folder:

```bash
python run_all_scenarios_v1.py --nrep 50 --nstarts 50
```

A quick test can be run with fewer replications:

```bash
python run_all_scenarios_v1.py --nrep 3 --scenarios S1 S5
```

The simulation outputs are written to:

```text
Code/sim_results/
```

This folder contains the CSV files, LaTeX table, boxplots, win-rate figure,
convergence figures, and regression-loss spread figures.

## Run the Real-Data Studies

For the diabetes data:

```bash
python realdata_runner.py --dataset diabetes --nrep 50 --nstarts 50 --k 2
```

For the gasoline NIR data:

```bash
python run_highdim_realdata.py --nrep 50 --nstarts 50 --k 2 --support_keep 60
```

The real-data outputs are written to:

```text
Code/realdata_results/
```

## Data

The diabetes data are loaded from `scikit-learn`.  
The gasoline NIR data are included in `Code/realdata/`.

## Settings Used in the Paper

The tuning parameters are fixed across replications to focus on the effect of
initialization:

```text
w = 0.5
lamV = 0.02
lamB = 0.01
rho1 = rho2 = rho3 = 2.0
```

For the low-dimensional simulation scenario S5, `lamV = 0.015` is used.

Both single-start and multi-start ADMM use the same inner ADMM updates and the
same stopping rule. The difference is only the initialization strategy.
