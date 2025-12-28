import numpy as np
from pyexpat.errors import XML_ERROR_BAD_CHAR_REF


def generate_data(
        n: int = 100,
        d: int = 3,
        noise_std: float = 0.1,
        seed: int = 42,
        ill_conditioned: bool = False,
):
    """
    Generate synthetic data for linear regression.
    -------
    n : Number of data points.
    d : Number of features.
    noise_std : Standard deviation of Gaussian noise.
    seed : Random seed.
    ill_conditioned : If true, generate ill-conditioned data (high collinearity)

    Returns
    -------
    X : ndarray, shape (n, d+1) Design matrix including bias term
    y : ndarray, shape (n,) Target values
    beta_true : ndarray, shape (d+1,) True regression coefficients (with bias)

    """


    rng = np.random.default_rng(seed)

    # --- true regression parameters (with bias) ---

    beta_true = np.array([1.0, 2.0, -1.5, 0.5]) # lenght = d+1

    # --- generate feature Matrix ---
    if not ill_conditioned:
        # well-contidtioned: independent features
        X_features = rng.normal(size=(n, d))
    else:
        # ill-conditioned: strong linear dependence
        x1 = rng.normal(size=(n, 1))
        X_features np.hstack(
            [
                x1,
                x1 + 1e-3 * rng.normal(size=x1.shape),
                x1 + 2e-3 * rng.normal(size=x1.shape),
            ]
        )

    # --- add bias colum ---
    X =  np.hstack([np.ones((n, 1)), X_features])

    # --- generate noisy targets ---
    noise = noise_std * rng.normal(size=n)
    y = X @ beta_true + noise

    return X, y, beta_true

if __name__ == "__main__":
    X_good, y_good, beta_good = generate_data(ill_conditioned=False)
    X_bad, y_bad, _ = generate_data(ill_conditioned=True)

    print("Well-conditioned X shape:", X_good.shape)
    print("Ill-conditioned X shape:", X_bad.shape)

    print("Condition number (good):", np.linalg.cond(X_good))
    print("Condition number (bad):", np.linalg.cond(X_bad))