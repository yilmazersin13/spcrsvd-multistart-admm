"""
demo_quickstart.py
Minimal single-start vs multi-start SPCRsvd comparison on a small synthetic
high-dimensional problem. Run: python demo_quickstart.py
"""

import os
import sys
import numpy as np
from numpy.linalg import svd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spcrsvd_core import (
    generate_dgp, standardize_from_train, center_y_from_train,
    ADMM_SPCR, multistart_ADMM, sinThetaF, rmse,
)


def main():
    rng = np.random.default_rng(0)
    n, p, k = 120, 300, 2

    X, y, B_true = generate_dgp(
        n=n, p=p, k=k, support_size=10,
        nuisance_rank=4, nuisance_strength=15.0, signal_strength=1.5,
        rng=rng,
    )

    idx = rng.permutation(n)
    ntr = int(0.7 * n)
    Xtr, Xte = standardize_from_train(X[idx[:ntr]], X[idx[ntr:]])
    ytr, yte = center_y_from_train(y[idx[:ntr]], y[idx[ntr:]])

    admm_kw = dict(w=0.5, lamV=0.02, lamB=0.01,
                   rho1=2.0, rho2=2.0, rho3=2.0,
                   max_iter=300, min_iter=50, support_keep=k * 10)

    _, _, Vt = svd(Xtr, full_matrices=False)
    fs = ADMM_SPCR(Xtr, ytr, Vt.T[:, :k], **admm_kw)
    fm = multistart_ADMM(Xtr, ytr, k=k, n_starts=30, rng=rng, **admm_kw)

    yhat_s = fs["beta0hat"] + Xte @ fs["Vhat"] @ fs["betahat"]
    yhat_m = fm["beta0hat"] + Xte @ fm["Vhat"] @ fm["betahat"]

    print("Test RMSE   single = %.4f   multi = %.4f" % (rmse(yte, yhat_s), rmse(yte, yhat_m)))
    print("sinTheta    single = %.4f   multi = %.4f"
          % (sinThetaF(fs["Vhat"], B_true), sinThetaF(fm["Vhat"], B_true)))
    print("Train Jreg  single = %.4f   multi = %.4f" % (fs["best_Jreg"], fm["best_Jreg"]))


if __name__ == "__main__":
    main()
