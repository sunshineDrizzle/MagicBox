import time
import numpy as np


def connectivity_grow(seeds_id, edge_list):
    """
    Find all vertices for each group of initial seeds.

    Parameters
    ----------
    seeds_id : list
        Its elements are also list, called sub-list, each sub-list
        contains a group of seed vertices which are used to
        initialize a evolving region.
        Different sub-list initializes different connected region.
    edge_list : dict | list
        The indices are vertices of a graph.
        One index's corresponding element is a collection of vertices
        which connect to the index.

    Return
    ------
    connected_regions : list
        Its elements are set, each set contains all vertices which
        connect to corresponding seeds.
    """
    connected_regions = [set(seeds) for seeds in seeds_id]

    for idx, region in enumerate(connected_regions):
        outmost_vtx = region.copy()
        while outmost_vtx:
            print(f'connected region{idx} size: {len(region)}')
            region_old = region.copy()
            for vtx in outmost_vtx:
                region.update(edge_list[vtx])
            outmost_vtx = region.difference(region_old)
    return connected_regions


def watershed(vtx2altitude, vtx2label, vtx2neighbors):
    """
    1. 找到所有种子区域的标记为0的一阶近邻，放进待合并列表里
    2. 从待合并列表里找出具有最小值的顶点，如果其近邻里有且仅有一个种子标记，
       将其合并进该种子区域；如果其近邻里有不止一个种子标记，将其标记为边界
       （值为-1）。然后把该顶点的标记为0的一阶近邻加入到待合并列表里。
    3. 重复第2步直到清空待合并列表。
    如果传入一个将所有局部最小值标记为不同种子区域的vtx2label就变成了
        原始的分水岭算法。

    Args:
        vtx2altitude (dict | sequence): topographic surface
            The keys/indices are vertices of a graph.
            One key/index's corresponding value/element is the altitude
                in the watershed algorithm.
        vtx2label (dict | sequence): marker mask
            The keys/indices are vertices of a graph.
            One key/index's corresponding value/element is a marker label.
                positive integer (e.g., 1, 2, 3): seed region
                0: unknown region
        vtx2neighbors (dict | sequence): edges
            The keys/indices are vertices of a graph.
            One key/index's corresponding value/element is a collection of
                vertices which connect to the key/index.

    Return:
        vtx2label (dict | sequence): marker mask
            The keys/indices are vertices of a graph.
            One key/index's corresponding value/element is a marker label.
                positive integer (e.g., 1, 2, 3): region
                -1: boundary
    """
    vtx_candidates = set()
    if isinstance(vtx2label, dict):
        for vtx, lbl in vtx2label.items():
            if lbl > 0:
                vtx_candidates.update(vtx2neighbors[vtx])
    else:
        vtx2label = np.asarray(vtx2label)
        seed_vertices = np.where(vtx2label > 0)[0]
        for seed_vtx in seed_vertices:
            vtx_candidates.update(vtx2neighbors[seed_vtx])
    vtx_candidates = [i for i in vtx_candidates if vtx2label[i] == 0]

    count = 0
    while vtx_candidates:
        time1 = time.time()

        altitudes = [vtx2altitude[i] for i in vtx_candidates]
        min_vtx = vtx_candidates[np.argmin(altitudes)]
        labels = []
        for neighbor in vtx2neighbors[min_vtx]:
            lbl = vtx2label[neighbor]
            labels.append(lbl)
            if lbl == 0 and (neighbor not in vtx_candidates):
                vtx_candidates.append(neighbor)
        labels = np.unique(labels)
        labels = labels[labels > 0]
        if len(labels) == 1:
            vtx2label[min_vtx] = labels[0]
        elif len(labels) == 0:
            raise RuntimeError('impossible')
        else:
            vtx2label[min_vtx] = -1
        vtx_candidates.remove(min_vtx)

        count += 1
        print(f'Finished {count}th iteration, cost {time.time()-time1} seconds.')
        print('Remained #candidates:', len(vtx_candidates))

    return vtx2label
