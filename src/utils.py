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