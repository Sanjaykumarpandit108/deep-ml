import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Step 1: Calculate transpose of X
    XT = X.T

    # Step 2: Calculate X^T * X
    XT_X = XT @ X

    # Step 3: Calculate inverse of X^T * X
    inverse = np.linalg.inv(XT_X)

    # Step 4: Calculate X^T * y
    XT_y = XT @ y

    # Step 5: Calculate theta
    theta = inverse @ XT_y

    # Round to 4 decimal places
    return np.round(theta, 4).tolist()