import numpy as np


def outlier_iqr(data, iqr_coef, axis=0):
    """
    Mark data points outside the [Q1-iqr_coefxIQR, Q3+iqr_coefxIQR]
    as outliers

    Args:
        data (array-like): 1D or 2D
        iqr_coef (float):
        axis (int): Default is 0
            Detect outliers alone the specified axis.

    Return:
        mask (bool array): outlier labels
            The shape of the mask is same as "data".
            Outliers are marked as True, others as False.
    """
    Q1 = np.percentile(data, 25, axis=axis, keepdims=True)
    Q3 = np.percentile(data, 75, axis=axis, keepdims=True)
    IQR = Q3 - Q1
    step = iqr_coef * IQR
    thr_l = Q1 - step
    thr_h = Q3 + step

    mask = np.zeros_like(data, dtype=bool)
    mask[data > thr_h] = True
    mask[data < thr_l] = True
    return mask


def outlier_z(data, z, axis=0):
    """
    Mark data points whose absolute values are larger that "z"
    as outliers

    Args:
        data (array-like): 1D or 2D
        z (float): outlier threshold value
            "z" is generally 3.0. As 99.7% of the data points
            lie between +/- 3 standard deviation 
            (using Gaussian Distribution approach).
        axis (int): Default is 0
            Detect outliers alone the specified axis.

    Return:
        mask (bool array): outlier labels
            The shape of the mask is same as "data".
            Outliers are marked as True, others as False.
    """
    pass
