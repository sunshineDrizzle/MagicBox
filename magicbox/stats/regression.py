import numpy as np
from sklearn.linear_model import LinearRegression


def regress_nuisance(X, Y, remove_cols=None):
    """
    For each regressor (a column of X), its mean is subtracted from its data.
    Each dependent variable (a column of Y) is then regressed against these,
    and a constant term. The resulting regressed slopes of all regressors
    specified with "remove_cols" are multiplied with their respective
    column data, and these are subtracted from each dependent variable.

    Args:
        X (2D array): regressors
            Each column is a independent variable.
        Y (2D array): dependent variables
            Each column is a dependent variable.
        remove_cols (sequence, optional): Defaults to None.
            Nuisance regressors' column indices in X
            If is None, all columns of X will be regarded as nuisance.

    Returns:
        Y_residual (2D array): residuals
            Each column is a residual of the corresponding dependent variable
            after removing the nuisance regressors.

    References:
        1. https://humanconnectome.org/software/workbench-command/-metric-regression
    """
    assert X.ndim == 2, 'X must be a 2D array!'
    assert Y.ndim == 2, 'Y must be a 2D array!'
    assert X.shape[0] == Y.shape[0]

    X = X - np.mean(X, 0, keepdims=True)
    reg = LinearRegression(fit_intercept=True).fit(X, Y)
    if remove_cols is None:
        Y_residual = Y - X.dot(reg.coef_.T)
    else:
        Y_residual = Y - X[:, remove_cols].dot(reg.coef_.T[remove_cols])

    return Y_residual
