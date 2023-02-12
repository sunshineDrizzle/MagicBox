import decimal
import numpy as np

from decimal import Decimal


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


def round_decimal(number, ndigits, round_type='half_up'):
    """
    Round a number to a given precision in decimal digits.

    Args:
        number (float): a float number
        ndigits (int): the number of decimal digits
            Only support positive integer at present
        round_type (str): Default is half_up
            half_up, floor, ceil

    Return:
        number (Decimal): a Decimal number after rounding
    """
    assert ndigits > 0 and isinstance(ndigits, int)
    number = Decimal(str(number))
    ctx = decimal.getcontext()
    if round_type == 'half_up':
        ctx.rounding = decimal.ROUND_HALF_UP
    elif round_type == 'floor':
        ctx.rounding = decimal.ROUND_FLOOR
    elif round_type == 'ceil':
        ctx.rounding = decimal.ROUND_CEILING
    else:
        raise ValueError('Not supported round type:', round_type)
    number = round(number, ndigits)

    return number


def smooth_1d(x, window='hanning', window_len=3):
    """
    Smooth the 1D data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing flipped copies of the signal
    (with the half of window size) in both ends so that transient parts are
    minimized in the begining and end part of the output signal.

    Args:
        x (1d array): the input signal
        window (str | 1d array-like): the type of window
            If it's a string, it's one of 'flat', 'hanning',
            'hamming', 'bartlett', 'blackman'. 'flat' window
            will produce a moving average smoothing.
            If it's a 1d array, it's a window in itself.
            'window_len' will be ignored.
        window_len (int): the dimension of the smoothing window;
            should be an odd integer

    Returns:
        y (1d array): the smoothed signal

    Examples:
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

    See also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
        numpy.convolve scipy.signal.lfilter

    References:
        1. https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html#smoothing-of-a-1d-signal
        2. https://www.delftstack.com/howto/python/smooth-data-in-python/
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("smooth_1d only accepts 1-dimensional arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be longer than window.")

    if isinstance(window, str):
        if window == 'flat':
            w = np.ones(window_len)
        elif window in ['hanning', 'hamming', 'bartlett', 'blackman']:
            w = eval('np.'+window+'(window_len)')
        else:
            raise ValueError("Window is one of flat, hanning, hamming,"
                             " bartlett, and blackman.")
    else:
        w = np.asarray(window)
        assert w.ndim == 1, "Window must be a 1d array."
        window_len = w.size

    assert window_len > 2, "Window length must be larger than 2."
    assert window_len % 2 == 1, "Window length must be an odd integer."

    half_len = int((window_len - 1) / 2)
    flip_idx1 = half_len - 0
    flip_idx2 = -2 - half_len
    s = np.r_[x[flip_idx1:0:-1], x, x[-2:flip_idx2:-1]]
    y = np.convolve(w/w.sum(), s, mode='valid')

    return y
