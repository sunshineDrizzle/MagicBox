from re import T
import numpy as np
from scipy.stats.stats import sem, zscore


def summary_across_col_by_mask(data, mask, values, metrics, tol_size=10, nan_mode=False,
                               row_names=None, zscore_flag=False, out_dict=False):
    """
    对2D数组每一行进行操作，将在mask中值相同的那些元素总结为一个标量。

    Args:
        data (2D array-like):
        mask (1D array-like): 长度和data的列数相同
        values (sequence): 指定使用mask中的哪些值
        metrics (str | strings): summary的方式
            'mean': 求均值
            'sem': 求标准误
            'sum': 求和
        tol_size (int, optional): Defaults to 10.
            可以容忍的样本量，如果用于总结的样本量小于该值，则输出警告
        nan_mode (bool, optional): Defaults to False.
            If True, 对每一行都检查并去掉NAN值
            If False, 默认所有元素都不是NAN
        row_names (sequence, optional): Defaults to None.
            行名
        zscore_flag (bool, optional): Defaults to False.
            If True, 在summary之前先对每一行做zscore
                如果nan_mode是True，是在去掉NAN之后做zscore
        out_dict (bool, optional): Defaults to False.
            If True, 返回的不是2D array，而是以行名为键，行为值的字典

    Returns:
        2D array | dict | list: 总结过后的数据
            如果metrics是str，返回单个2D array或dict
            如果metrics是strings，返回一列表的2D array或dict
            返回的是2D array还是dict，参见out_dict的参数说明
    """
    # prepare data
    data = np.asarray(data)
    assert data.ndim == 2
    n_row, n_col = data.shape

    # prepare mask
    mask = np.asarray(mask)
    assert mask.shape == (n_col,)
    n_v = len(values)

    # prepare metric
    metrics_str_flag = False
    if isinstance(metrics, str):
        metrics = [metrics]
        metrics_str_flag = True
    metric2func = {
        'mean': np.mean,
        'sem': sem,
        'sum': np.sum
    }

    # prepare row names
    if row_names is None:
        row_names = np.arange(n_row)
    else:
        assert len(row_names) == n_row

    # calculating
    out_data = [np.zeros((n_row, n_v), np.float64) for _ in metrics]
    if nan_mode:
        for row_idx, row in enumerate(data):
            non_nan_vec = ~np.isnan(row)
            row = row[non_nan_vec]
            if zscore_flag:
                row = zscore(row)
            mask_tmp = mask[non_nan_vec]
            for v_idx, v in enumerate(values):
                samples = row[mask_tmp == v]
                n_sample = len(samples)
                if n_sample < tol_size:
                    print(f'Warning! The sample size of value-{v} in '
                          f'row-{row_names[row_idx]} is {n_sample}.')
                for metric_idx, metric in enumerate(metrics):
                    out_data[metric_idx][row_idx, v_idx] = metric2func[metric](samples)
    else:
        if zscore_flag:
            data = zscore(data, 1)
        for v_idx, v in enumerate(values):
            mask_tmp = mask == v
            n_sample = np.sum(mask_tmp)
            if n_sample < tol_size:
                print(f'Warning! The sample size of value-{v} is {n_sample}.')
            for metric_idx, metric in enumerate(metrics):
                out_data[metric_idx][:, v_idx] = \
                    metric2func[metric](data[:, mask_tmp], axis=1)

    # return
    if out_dict:
        for i, e in enumerate(out_data):
            e_new = dict()
            for row_idx, row in enumerate(e):
                e_new[row_names[row_idx]] = row
            out_data[i] = e_new
    if metrics_str_flag:
        out_data = out_data[0]
    return out_data
