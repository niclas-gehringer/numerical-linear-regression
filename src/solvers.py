import numpy as np

def _validate_xy(X: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    """Validate shapes/types for least squares solver and return (n,p)"""
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError(f"X and y incompatible: X has {n} rows, y has {y.shape[0]} entries")

    return (n, p)


def solve_qr(X,y)-> np.ndarray:
    """
        Solve the least squares problem min ||X beta - y||_2 using QR decomposition.


    Inputs
    -------
    X : ndarray, shape (n, d+1)
        Design matrix including bias term
    y : ndarray, shape (n,)
        Target values

    Returns
    -------
    beta_hat : ndarray, shape (p,)
    Least squares solution (QR-based)
    """
    # --- basic validation ---
    n, p = _validate_xy(X, y)

    # --- compute the QR decomposition of X ---
    Q, R = np.linalg.qr(X, mode="reduced")

    # --- transform right side ---
    z = Q.T @ y

    # --- Solve the resulting system of equations ---
    beta_hat = np.linalg.solve(R, z)

    return beta_hat


def solve_svd(X, y, tol: float | None = None)-> np.ndarray:
    """
    Solve the least squares problem min ||X beta - y||_2 using SVD.

    Inputs
    -------
    X : ndarray, shape (n, d+1)
        Design matrix including bias term
    y : ndarray, shape (n,)
        Target values
    tol : float | None
    Singular values <= tol are treated as zero (for numerical stability).
        If None, a standard default is used: tol = max(n, p) * eps * sigma_max.

    Returns
    -------
    beta_hat : ndarray, shape (p,)
        Least squares solution (SVD-based, with thresholding).
    """
    # --- basic validation ---
    n, p = _validate_xy(X, y)

    # --- SVD: X = U Σ V^T ---
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # --- choose Tolerance if not provided ---
    if tol is None:
        eps = np.finfo(X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64).eps
        tol = max(n, p) * eps * s[0]

    # --- build Σ^+ implicitly via reciprocal of singular values (with threshold) ---
    s_inv = np.zeros_like(s)
    mask = s > tol
    s_inv[mask] = 1.0 / s[mask]

    # --- beta = V Σ^+ U^T y ---
    # Stepwise:
    # z = U^T y          (shape r,)
    # w = Σ^+ z          (shape r,) => elementwise multiply by s_inv
    # beta = V w         (shape p,) where V = Vt^T
    z = U.T @ y
    w = s_inv * z
    beta_hat = Vt.T @ w

    return beta_hat



