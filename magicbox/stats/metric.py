import numpy as np

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm, AnovaRM


class ANOVA:
    """
    Methods:
    -------
    eta_squared, omega_squared:
        As a Psychologist most of the journals we publish in requires to report effect sizes.
        Common software, such as, SPSS have eta squared as output.
        However, eta squared is an overestimation of the effect.
        To get a less biased effect size measure we can use omega squared.
        The two methods adds eta squared and omega squared to the DataFrame that contains the ANOVA table.

    References:
    ----------
    1. http://www.pybloggers.com/2016/03/three-ways-to-do-a-two-way-anova-with-python/
    2. https://pythonfordatascience.org/anova-2-way-n-way/
    3. https://www.marsja.se/four-ways-to-conduct-one-way-anovas-using-python/
    4. http://www.pybloggers.com/2018/10/repeated-measures-anova-in-python-using-statsmodels/
    5. https://www.marsja.se/repeated-measures-anova-in-python-using-statsmodels/
    """
    def one_way(self, data, dep_var, factor):
        """
        one-way ANOVA

        Parameters:
        ----------
        data: DataFrame
            Contains at least 2 columns that are 'dependent variable' and 'factor' respectively.
        dep_var: str
            Name of the 'dependent variable' column.
        factor: str
            Name of the 'factor' column.

        Return:
        ------
        aov_table: DataFrame
            ANOVA table
        """
        formula = '{} ~ {}'.format(dep_var, factor)
        print('formula:', formula)
        model = ols(formula, data).fit()
        aov_table = anova_lm(model, typ=2)
        self.eta_squared(aov_table)
        self.omega_squared(aov_table)

        return aov_table

    def two_way(self, data, dep_var, factor1, factor2):
        """
        two-way ANOVA

        Parameters:
        ----------
        data: DataFrame
            Contains at least 3 columns that are 'dependent variable', 'factor1', and 'factor2' respectively.
        dep_var: str
            Name of the 'dependent variable' column.
        factor1: str
            Name of the 'factor1' column.
        factor2: str
            Name of the 'factor2' column.

        Return:
        ------
        aov_table: DataFrame
            ANOVA table
        """
        formula = '{0} ~ C({1}) + C({2}) + C({1}):C({2})'.format(dep_var, factor1, factor2)
        print('formula:', formula)
        model = ols(formula, data).fit()
        aov_table = anova_lm(model, typ=2)
        self.eta_squared(aov_table)
        self.omega_squared(aov_table)

        return aov_table

    def rm(self, data, dep_var, subject, within, aggregate_func=None):
        """
        Repeated Measures ANOVA

        Parameters:
        ----------
        data: DataFrame
            Contains at least 3 columns that are 'dependent variable', 'subject', and 'factor' respectively.
        dep_var: str
            Name of the 'dependent variable' column.
        subject: str
            Name of the 'subject' column. (subject identifier)
        within: a list of strings
            Names of the at least one 'factor' columns.

        Return:
        ------
        aov_table: DataFrame
            ANOVA table
        """
        aov_rm = AnovaRM(data, dep_var, subject, within, aggregate_func=aggregate_func)
        aov_table = aov_rm.fit().anova_table

        return aov_table

    @staticmethod
    def eta_squared(aov):
        aov['eta_sq'] = 'NaN'
        aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
        return aov

    @staticmethod
    def omega_squared(aov):
        mse = aov['sum_sq'][-1] / aov['df'][-1]
        aov['omega_sq'] = 'NaN'
        aov['omega_sq'] = (aov[:-1]['sum_sq'] - (aov[:-1]['df'] * mse)) / (sum(aov['sum_sq']) + mse)
        return aov


class EffectSize:

    def cohen_d(self, sample1, sample2):
        """
        Calculate Cohen's d.
        如果其中一个样本A是只包含一个0值的序列,
            那就是计算另一个样本B相对于0的Cohen's d (对应单样本t检验)
            如果此时另一个样本B实际上是两个配对样本的差,
            那就是计算这两个配对样本之间的Cohen's d (对应配对t检验)
                d = np.mean(B) / np.std(B, ddof=1)
        其它情况就是计算这两个样本之间的Cohen's d (对应双样本t检验)
            d = (np.mean(sample1) - np.mean(sample2)) / s_pool

        Parameters:
        ----------
        sample1: array-like with one dimension
        sample2: array-like with one dimension

        Return:
        ------
        d: float
            the value of the Cohen's d between sample1 and sample2

        References:
        ----------
        1. https://machinelearningmastery.com/effect-size-measures-in-python/
        2. https://www.statisticshowto.datasciencecentral.com/cohens-d/
        3. https://stackoverflow.com/questions/21532471/how-to-calculate-cohens-d-in-python
        4. https://www.real-statistics.com/students-t-distribution/paired-sample-t-test/cohens-d-paired-samples/
        5. https://www.sohu.com/a/168329069_489312
        """
        # calculate the size of samples
        n1, n2 = len(sample1), len(sample2)

        # calculate the variance of the samples
        # the divisor used in the calculation is n1, n2 respectively.
        v1, v2 = np.var(sample1), np.var(sample2)

        # calculate the pooled standard deviation
        s = np.sqrt((n1 * v1 + n2 * v2) / (n1 + n2 - 2))

        # calculate the means of the samples
        u1, u2 = np.mean(sample1), np.mean(sample2)

        # calculate the effect size
        d = (u1 - u2) / s
        return d


def calc_coef_var(arr, axis=None, ddof=0):
    """
    改进后的变异系数(coefficient of variation, CV)计算方式
    计算作为分母的均值之前先取绝对值（标准差还是基于原数据计算）

    Args:
        arr (array-like):
        axis (int, optional): Defaults to None.
            calculate CV along which axis
        ddof (int, optional): Defaults to 0.

    Returns:
        [float]: coefficient of variation
    """
    var = np.std(arr, axis, ddof=ddof) /\
        np.mean(np.abs(arr), axis)
    return var


def calc_cqv(arr, axis=None):
    """
    Calculate coefficient of quartile variation
    https://en.wikipedia.org/wiki/Quartile_coefficient_of_dispersion

    Args:
        arr (array-like):
        axis (int, optional): Defaults to None.
            calculate CQV along which axis

    Returns:
        [float]: coefficient of quartile variation
    """
    q1 = np.percentile(arr, 25, axis)
    q3 = np.percentile(arr, 75, axis)
    var = (q3 - q1) / (q3 + q1)
    return var


def _overlap(c1, c2, index='dice'):
    """
    Calculate overlap between two collections

    Parameters
    ----------
    c1, c2 : collection (list | tuple | set | 1-D array etc.)
    index : string ('dice' | 'percent')
        the index used to measure overlap

    Return
    ------
    overlap : float
        The overlap between c1 and c2
    """
    set1 = set(c1)
    set2 = set(c2)
    intersection_num = float(len(set1 & set2))
    try:
        if index == 'dice':
            total_num = len(set1 | set2) + intersection_num
            overlap = 2.0 * intersection_num / total_num
        elif index == 'percent':
            overlap = 1.0 * intersection_num / len(set1)
        else:
            raise Exception("Unsupported index:", index)
    except ZeroDivisionError as e:
        print(e)
        overlap = np.nan
    return overlap


def calc_overlap(data1, data2, label1=None, label2=None, index='dice'):
    """
    Calculate overlap between two sets.
    The sets are acquired from data1 and data2 respectively.

    Parameters
    ----------
    data1, data2 : collection or numpy array
        label1 is corresponding with data1
        label2 is corresponding with data2
    label1, label2 : None or labels
        If label1 or label2 is None, the corresponding data is supposed to be
            a collection of members such as vertices and voxels.
        If label1 or label2 is a label, the corresponding data is always a
            numpy array with same shape and meaning. And we will acquire set1
            elements whose labels are equal to label1 from data1 and set2
            elements whose labels are equal to label2 from data2.
    index : string ('dice' | 'percent')
        the index used to measure overlap

    Return
    ------
    overlap : float
        The overlap of data1 and data2
    """
    if label1 is not None:
        positions1 = np.where(data1 == label1)
        data1 = list(zip(*positions1))

    if label2 is not None:
        positions2 = np.where(data2 == label2)
        data2 = list(zip(*positions2))

    # calculate overlap
    overlap = _overlap(data1, data2, index)

    return overlap


def loocv_overlap(X, prob, metric='dice'):
    """
    Calculate overlaps for leave-one-out cross validation.
    Each sample has its own region of interest (ROI). For each iteration,
    overlap between the ROI in the left sample and the ROI in remaining samples
    will be calculated. The ROI in remaining samples is defined as below:
        Calculate probability map for the remaining samples, regard locations
        whose probability is suprathreshold as the ROI.

    Parameters:
    ----------
    X[ndarray]: shape=(n_sample, n_location)
        Its data type must be bool. Each row is a sample.
        Each sample's region of interest consists of the locations with True values.
    prob[float]: the threshold probability
    metric[str]: string ('dice' | 'percent')
        Specify a metric which is used to measure overlap.

    Return:
    ------
    overlaps[ndarray]: shape=(n_sample,)
    """
    assert X.ndim == 2, 'The input X must be a 2D array!'
    assert X.dtype == np.bool, "The input X's data type must be bool!"
    n_samp, _ = X.shape

    remain_idx_arr = np.ones((n_samp,), dtype=np.bool)
    overlaps = np.zeros((n_samp,))
    for left_idx in range(n_samp):
        # get roi of the left sample
        roi_left = np.where(X[left_idx])[0]

        # get roi of the remaining samples
        remain_idx_arr[left_idx] = False
        prob_map = np.mean(X[remain_idx_arr], 0)
        roi_remain = np.where(prob_map > prob)[0]
        remain_idx_arr[left_idx] = True

        # calculate overlap
        overlaps[left_idx] = calc_overlap(roi_left, roi_remain, index=metric)

    return overlaps
