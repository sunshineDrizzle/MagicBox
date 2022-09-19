from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from matplotlib import pyplot as plt


def hac_scipy(data, cluster_nums, method='ward', metric='euclidean',
              criterion='maxclust', optimal_ordering=False, out_path=None):
    """
    Perform hierarchical/agglomerative clustering on data

    Parameters
    ----------
    data: see linkage
    cluster_nums: sequence | iterator
        Each element is the number of clusters that HAC generate.
    method: see linkage
    metric: see linkage
    criterion: see fcluster
    optimal_ordering: see linkage
    out_path: str
        plot hierarchical clustering as a dendrogram and
        save it out to the path

    Return
    ------
    labels_list: list
        label results of each cluster_num
    """
    # do hierarchical clustering on data
    Z = linkage(data, method, metric, optimal_ordering)
    labels_list = []
    for num in cluster_nums:
        labels_list.append(fcluster(Z, num, criterion))
        print('HAC finished: {}'.format(num))

    if out_path is not None:
        # show the dendrogram
        plt.figure()
        dendrogram(Z)
        plt.savefig(out_path)

    return labels_list
