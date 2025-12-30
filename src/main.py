import numpy as np
import solvers
import data
from typing import Optional

def train_test_split(X, y, train_ratio = 0.8):
    """
        Split the generated data into train and test sets
        -------
        X : ndarray, shape (n, d+1) Design matrix including bias term
        y : ndarray, shape (n,) Target values

        Returns
        -------
        X_train : ndarray, shape (0.8 * n, d)
        X_test : ndarray, shape (0.2 * n, d)

        y_train : ndarray, shape (0.8 * n,) Target values
        y_test : ndarray, shape (0.2 * n,) Target values

        """

    n = X.shape[0]
    k = int(train_ratio * n)

    X_train = X[:k]
    X_test = X[k:]
    y_train = y[:k]
    y_test = y[k:]

    # --- check dimensions --- #
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train has {X_train.shape[0]} rows, but y_train has {y_train.shape[0]} entries")

    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"X_train has {X_test.shape[0]} rows, but y_train has {y_test.shape[0]} entries")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"Train/test feature mismatch: expected same number of columns, got {X_train.shape[1]} (train) and {X_test.shape[1]} (test).")

    return X_train, X_test, y_train, y_test

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio = train_ratio)

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
    seed = 43               # reproducibility seed
    svd_tol = 1e-1            # SVD tolerance handling

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