import numpy as np

from scipy.spatial.distance import cdist, pdist


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


# >>>variation
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
# variation<<<


# >>>overlap
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
# overlap<<<


# >>>cluster
def elbow_score(X, labels, metric='euclidean', type=('inner', 'standard')):
    """
    calculate elbow score for a partition specified by labels
    https://en.wikipedia.org/wiki/Elbow_method_(clustering)

    :param X: array, shape = (n_samples, n_features)
        a feature array
    :param labels: array, shape = (n_samples,)
        Predicted labels for each sample.
    :param metric: string
        Specify how to calculate distance between samples in a feature array.
        Options: 'euclidean', 'correlation'
    :param type: tuple of two strings
        Options:
        ('inner', 'standard') - Implement Wk in (Tibshirani et al., 2001b)
        ('inner', 'centroid') - For each cluster, calculate the metric between samples within it
                                with the cluster's centroid. Finally, average all samples.
        ('inner', 'pairwise') - For each cluster, calculate the metric pairwise among samples within it.
                                Finally, average all samples.
        ('inter', 'centroid') - Calculate the metric between cluster centroids with their centroid.
                                Finally, average all clusters.
        ('inter', 'pairwise') - Calculate the metric pairwise among cluster centroids.
                                Finally, average all clusters.

    :return: score:
        elbow score of the partition
    """
    if type == ('inner', 'standard'):
        score = 0
        for label in set(labels):
            sub_samples = np.atleast_2d(X[labels == label])
            dists = cdist(sub_samples, sub_samples, metric=metric)
            tmp_score = np.sum(dists) / (2.0 * sub_samples.shape[0])
            score += tmp_score
    elif type == ('inner', 'centroid'):
        # https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion
        # formula-1 in (Goutte, Toft et al. 1999 - NeuroImage)
        sub_scores = []
        for label in set(labels):
            sub_samples = np.atleast_2d(X[labels == label])
            sub_samples_centroid = np.atleast_2d(np.mean(sub_samples, 0))
            tmp_scores = cdist(sub_samples_centroid, sub_samples, metric=metric)[0]
            sub_scores.extend(tmp_scores)
        score = np.mean(sub_scores)
    elif type == ('inner', 'pairwise'):
        sub_scores = []
        for label in set(labels):
            sub_samples = np.atleast_2d(X[labels == label])
            sub_scores.extend(pdist(sub_samples, metric=metric))
        score = np.mean(sub_scores)
    elif type == ('inter', 'centroid'):
        # adapted from formula-2 in (Goutte, Toft et al. 1999 - NeuroImage)
        sub_centroids = []
        for label in set(labels):
            sub_samples = np.atleast_2d(X[labels == label])
            sub_centroids.append(np.mean(sub_samples, 0))
        centroid = np.atleast_2d(np.mean(sub_centroids, 0))
        tmp_scores = cdist(centroid, np.array(sub_centroids), metric=metric)[0]
        score = np.mean(tmp_scores)
    elif type == ('inter', 'pairwise'):
        sub_centroids = []
        for label in set(labels):
            sub_samples = np.atleast_2d(X[labels == label])
            sub_centroids.append(np.mean(sub_samples, 0))
        sub_centroids = np.array(sub_centroids)
        if sub_centroids.shape[0] == 1:
            sub_centroids = np.r_[sub_centroids, sub_centroids]
        score = np.mean(pdist(sub_centroids, metric=metric))
    else:
        raise TypeError('Type-{} is not supported at present.'.format(type))

    return score


def gap_statistic(X, cluster_nums, ref_num=10, cluster_method=None):
    """
    do clustering with gap statistic assessment according to (Tibshirani et al., 2001b)
    https://blog.csdn.net/baidu_17640849/article/details/70769555
    https://datasciencelab.wordpress.com/tag/gap-statistic/
    https://github.com/milesgranger/gap_statistic

    :param X: array, shape = (n_samples, n_features)
        a feature array
    :param cluster_nums: a iterator of integers
        Each integer is the number of clusters to try on the data.
    :param ref_num: integer
        The number of random reference data sets used as inertia reference to actual data.
    :param cluster_method: callable
        The cluster method to do clustering on the feature array. And the method returns
        labels_list (cluster results of each cluster_num in cluster_nums).
        If is None, a default K-means method will be used.

    :return: labels_list: list
        cluster results of each cluster_num in cluster_nums
    :return: Wks: array, shape = (len(cluster_nums),)
        within-cluster dispersion of each cluster_num clustering on the feature array X
    :return: Wks_refs_log_mean: array, shape = (len(cluster_nums),)
        mean within-cluster dispersion of each cluster_num clustering on ref_num reference data sets
    :return: gaps: array, shape = (len(cluster_nums),)
        Wks_refs_log_mean - np.log(Wks)
    :return: s: array, shape = (len(cluster_nums),)
        I think elements in s can be regarded as standard errors of gaps.
    :return: k_selected: integer
        cluster k_selected clusters on X may be the best choice
    """
    if cluster_method is None:
        def k_means(data, cluster_nums):
            """
            http://scikit-learn.org/stable/modules/clustering.html#k-means
            """
            from sklearn.cluster import KMeans

            labels_list = []
            for cluster_num in cluster_nums:
                kmeans = KMeans(cluster_num, random_state=0, n_init=10).fit(data)
                labels_list.append(kmeans.labels_ + 1)
                print('KMeans finished: {}'.format(cluster_num))
            return labels_list

        cluster_method = k_means

    print('Start: calculate W\u2096s')
    Wks = []
    labels_list = cluster_method(X, cluster_nums)
    for labels in labels_list:
        Wks.append(elbow_score(X, labels))
    Wks = np.array(Wks)
    Wks_log = np.log(Wks)
    print('Finish: calculate W\u2096s')

    print("Start: calculate references' W\u2096s")
    Wks_refs_log = []
    minimums = np.atleast_2d(np.min(X, axis=0))
    maximums = np.atleast_2d(np.max(X, axis=0))
    bounding_box = np.r_[minimums, maximums]
    for i in range(ref_num):
        X_ref = uniform_box_sampling(X.shape[0], bounding_box)
        labels_list_ref = cluster_method(X_ref, cluster_nums)
        Wks_ref_log = []
        for labels in labels_list_ref:
            Wks_ref_log.append(np.log(elbow_score(X_ref, labels)))
        Wks_refs_log.append(Wks_ref_log)
        print('Finish reference: {}/{}'.format(i+1, ref_num))
    print("Finish: calculate references' W\u2096s")

    print('Start: calculate gaps')
    Wks_refs_log = np.array(Wks_refs_log)
    Wks_refs_log_mean = np.mean(Wks_refs_log, axis=0)
    Wks_refs_log_std = np.std(Wks_refs_log, axis=0)
    gaps = Wks_refs_log_mean - Wks_log
    print('Finish: calculate gaps')

    print('Start: select optimal k')
    s = Wks_refs_log_std * np.sqrt(1 + 1.0 / ref_num)
    idx_selected = np.where(gaps[:-1] >= gaps[1:] - s[1:])[0][0]
    k_selected = cluster_nums[idx_selected]
    print('Finish: select optimal k')

    return labels_list, Wks, Wks_refs_log_mean, gaps, s, k_selected
# cluster<<<
