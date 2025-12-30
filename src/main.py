import numpy as np
import solvers
import data
from typing import Optional
import utils

def run_experiment(
        label: str,
        n: int,
        d: int,
        noise_std: float,
        seed: int,
        train_ratio: float,
        ill_conditioned: bool,
        tol: Optional[float] = None
):
    """
       Run a single train/test experiment for linear regression and
       compare QR and SVD solvers.

       Parameters
       ----------
       label : str
           Label used for printing results.
       n : int
           Number of data points.
       d : int
           Number of features (excluding bias).
       noise_std : float
           Standard deviation of Gaussian noise.
       seed : int
           Random seed for reproducibility.
       train_ratio : float
           Fraction of data used for training.
       ill_conditioned : bool
           Whether to generate ill-conditioned data.

    """

    # --- 1) Generate synthetic data ---
    X, y, beta_true = data.generate_data(
        n=n,
        d=d,
        noise_std=noise_std,
        seed=seed,
        ill_conditioned=ill_conditioned,
    )
    cond_X = np.linalg.cond(X)
    print(f"Condition number of X: {cond_X:.4e}")

    # --- 2) train / test split ---
    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, train_ratio = train_ratio)

    # --- 3) Apply models (QR and SVD) ---
    beta_qr = solvers.solve_qr(X_train, y_train)
    if tol is None:
        beta_svd = solvers.solve_svd(X_train, y_train)
    else: beta_svd = solvers.solve_svd(X_train, y_train, tol=tol)

    # --- 4) Predict on test data ---
    y_pred_qr = X_test @ beta_qr
    y_pred_svd = X_test @ beta_svd

    # --- 5) Evaluate ---
    rmse_qr = np.sqrt(np.mean((y_pred_qr - y_test)**2))
    rmse_svd = np.sqrt(np.mean((y_pred_svd - y_test)**2))

    beta_err_qr = np.linalg.norm(beta_qr - beta_true)
    beta_err_svd = np.linalg.norm(beta_svd - beta_true)

    # --- Print results ---
    print(f"\n[{label}]")
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"RMSE (QR) : {rmse_qr:.4e}")
    print(f"RMSE (SVD): {rmse_svd:.4e}")
    print(f"Beta error (QR) : {beta_err_qr:.4e}")
    print(f"Beta error (SVD): {beta_err_svd:.4e}")

    print("\nExample predictions (y_true | y_qr | y_svd):")
    for i in range(min(3, len(y_test))):
        print(f"{y_test[i]: .4f} | {y_pred_qr[i]: .4f} | {y_pred_svd[i]: .4f}")

if __name__ == "__main__":
    # --- Experiment configuration ---
    n = 200                 # number of data points
    d = 3                   # number of features (excluding bias)
    noise_std = 0.1         # light Gaussian noise
    train_ratio = 0.8       # 80/20 split
    seed = 42               # reproducibility seed
    svd_tol = 1e-1          # SVD tolerance handling

    # --- Run both scenarios ---
    print("\n=== Well-conditioned dataset ===")
    run_experiment(
        label="Well-conditioned",
        n=n,
        d=d,
        noise_std=noise_std,
        seed=seed,
        train_ratio=train_ratio,
        ill_conditioned=False,
        tol=svd_tol
    )

    print("\n=== Ill-conditioned dataset ===")
    run_experiment(
        label="Ill-conditioned",
        n=n,
        d=d,
        noise_std=noise_std,
        seed=seed+1,
        train_ratio=train_ratio,
        ill_conditioned=True,
        tol=svd_tol
    )