import numpy as np
from scipy.stats import pearsonr
from .outlier import outlier_iqr


def calc_pearson_r_p(data1, data2, nan_mode=False, iqr_coef=None):
    """
    data1的形状是m1 x n, data2的形状是m2 x n
    用data1的每一行和data2的每一行做皮尔逊相关，得到：
    m1 x m2的r矩阵和p矩阵

    if nan_mode is True:
        每两行做相关之前会去掉值为NAN的样本点
    if iqr_coef is not None:
        每两行做相关之前会去掉值在iqr_coef倍IQR以外的样本点
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    assert data1.ndim == 2 and data2.ndim == 2
    m1, n = data1.shape
    m2, n2 = data2.shape
    assert n == n2

    r_arr = np.zeros((m1, m2), np.float64)
    p_arr = np.zeros((m1, m2), np.float64)
    if nan_mode:
        non_nan_arr1 = ~np.isnan(data1)
        non_nan_arr2 = ~np.isnan(data2)
        for i in range(m1):
            for j in range(m2):
                non_nan_vec = np.logical_and(non_nan_arr1[i], non_nan_arr2[j])
                if np.sum(non_nan_vec) < 2:
                    r, p = np.nan, np.nan
                else:
                    x = data1[i][non_nan_vec]
                    y = data2[j][non_nan_vec]
                    if iqr_coef is None:
                        r, p = pearsonr(x, y)
                    else:
                        outlier1 = outlier_iqr(x, iqr_coef)
                        outlier2 = outlier_iqr(y, iqr_coef)
                        outlier_mask = np.logical_or(outlier1, outlier2)
                        mask = ~outlier_mask
                        if np.sum(mask) < 2:
                            r, p = np.nan, np.nan
                        else:
                            x = x[mask]
                            y = y[mask]
                            r, p = pearsonr(x, y)
                r_arr[i, j] = r
                p_arr[i, j] = p
    else:
        if iqr_coef is not None:
            outlier_arr1 = outlier_iqr(data1, iqr_coef, 1)
            outlier_arr2 = outlier_iqr(data2, iqr_coef, 1)
        for i in range(m1):
            for j in range(m2):
                x = data1[i]
                y = data2[j]
                if iqr_coef is not None:
                    outlier_mask = np.logical_or(
                        outlier_arr1[i], outlier_arr2[j])
                    mask = ~outlier_mask
                    x = x[mask]
                    y = y[mask]
                r, p = pearsonr(x, y)
                r_arr[i, j] = r
                p_arr[i, j] = p

    return r_arr, p_arr
