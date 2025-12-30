import os
import matplotlib.pyplot as plt
import numpy as np

import data
import solvers
import utils

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def run_tolerance_sweep(
    n: int = 200,
    d: int = 3,
    noise_std: float = 0.1,
    seed: int = 42,
    train_ratio: float = 0.8,
):
    # --- Generate one ill-conditioned dataset (fixed) ---
    X, y, beta_true = data.generate_data(
        n=n,
        d=d,
        noise_std=noise_std,
        seed=seed,
        ill_conditioned=True,
    )

    X_train, X_test, y_train, y_test = utils.train_test_split(X,y, train_ratio=train_ratio)

    # --- QR baseline (independent of tol) ---
    beta_qr = solvers.solve_qr(X_train, y_train)
    y_pred_qr = X_test @ beta_qr
    qr_beta_norm = float(np.linalg.norm(beta_qr))
    qr_rmse = rmse(y_test, y_pred_qr)

    # --- Choose tol grid (log-spaced, plus None baseline for SVD) ---
    # We'll evaluate SVD for  these tolerances
    tol_values = [None] + [10.0 ** (-k) for k in range(12, 0, -1)] # 1e-12 ... 12-1

    svd_beta_norm = []
    svd_rmses = []
    tol_numeric = [] # for plotting on a log x-axis

    for tol in tol_values:
        if tol is None:
            beta_svd = solvers.solve_svd(X_train,y_train)
            y_pred_svd = X_test @ beta_svd
            default_svd_beta_norm = float(np.linalg.norm(beta_svd))
            default_svd_rmse = rmse(y_test, y_pred_svd)
            continue

            continue

        beta_svd = solvers.solve_svd(X_train,y_train, tol=float(tol))
        y_pred_svd = X_test @ beta_svd

        svd_beta_norm.append(float(np.linalg.norm(beta_svd)))
        svd_rmses.append(rmse(y_test, y_pred_svd))
        tol_numeric.append(float(tol))

    return {
        "cond_X": float(np.linalg.cond(X)),
        "qr_beta_norm": qr_beta_norm,
        "qr_rmse": qr_rmse,
        "default_svd_beta_norm": default_svd_beta_norm,
        "default_svd_rmse": default_svd_rmse,
        "tol": np.array(tol_numeric),
        "svd_beta_norm": np.array(svd_beta_norm),
        "svd_rmse": np.array(svd_rmses),
    }

def save_plots(results: dict, out_dir: str = "figures"):
    os.makedirs(out_dir, exist_ok=True)

    tol = results["tol"]

    # Plot 1: ||beta|| vs tol
    plt.figure()
    plt.xscale("log")
    plt.plot(tol, results["svd_beta_norm"], marker="o", linestyle="-", label="SVD (truncated)")
    plt.axhline(results["qr_beta_norm"], linestyle="--", label="QR (baseline)")
    plt.title(f"Ill-conditioned: Coefficient norm vs tolerance (cond(X)≈{results['cond_X']:.2e})")
    plt.xlabel("SVD tolerance (tol)")
    plt.ylabel("||beta||_2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "beta_norm_vs_tol.png"), dpi=200)
    plt.close()

    # Plot 2: Test RMSE vs tol
    plt.figure()
    plt.xscale("log")
    plt.plot(tol, results["svd_rmse"], marker="o", linestyle="-", label="SVD (truncated)")
    plt.axhline(results["qr_rmse"], linestyle="--", label="QR (baseline)")
    plt.title(f"Ill-conditioned: Test RMSE vs tolerance (cond(X)≈{results['cond_X']:.2e})")
    plt.xlabel("SVD tolerance (tol)")
    plt.ylabel("RMSE (test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_vs_tol.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
        results = run_tolerance_sweep(
            n=200,
            d=3,
            noise_std=0.1,
            seed=42,
            train_ratio=0.8,
        )

        print(f"cond(X) = {results['cond_X']:.4e}")
        print(f"QR:      ||beta||={results['qr_beta_norm']:.4e}, RMSE={results['qr_rmse']:.4e}")
        print(f"SVD(def):||beta||={results['default_svd_beta_norm']:.4e}, RMSE={results['default_svd_rmse']:.4e}")

        save_plots(results, out_dir="figures")
        print("Saved figures to ./figures/")

