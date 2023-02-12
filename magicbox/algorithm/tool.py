import numpy as np


# --------sampling--------
def uniform_box_sampling(n_sample, bounding_box=((0,), (1,))):
    """
    create n_sample samples with uniform distribution in the box
    https://blog.csdn.net/baidu_17640849/article/details/70769555
    https://datasciencelab.wordpress.com/tag/gap-statistic/

    :param n_sample: integer
        the number of samples
    :param bounding_box: array-like, shape = (2, n_dim)
        Shape[1] is the number of dimensions.
        Bounding_box[0] are n_dim minimums of their own dimensions.
        Bounding_box[1] are n_dim maximums of their own dimensions.

    :return: samples: array, shape = (n_sample, n_dim)
    """
    bounding_box = np.array(bounding_box)
    dists = np.diag(bounding_box[1] - bounding_box[0])
    samples = np.random.random_sample((n_sample, bounding_box.shape[1]))
    samples = np.matmul(samples, dists) + bounding_box[0]

    return samples


# ---common---
def intersect(arr, mask, label=None, substitution=np.nan):
    """
    reserve values in the mask and replace values out of the mask with substitution
    :param arr: numpy array
    :param mask: numpy array
    :param label:
        specify the mask value in the mask array
    :param substitution:
    :return:
    """
    assert arr.shape == mask.shape

    if label is None:
        mask_idx_mat = mask != 0
    else:
        mask_idx_mat = mask == label

    if substitution == 'min':
        substitution = np.min(arr[mask_idx_mat])
    elif substitution == 'max':
        substitution = np.max(arr[mask_idx_mat])

    new_arr = arr.copy()
    new_arr[np.logical_not(mask_idx_mat)] = substitution
    return new_arr
