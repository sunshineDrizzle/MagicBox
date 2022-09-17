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
