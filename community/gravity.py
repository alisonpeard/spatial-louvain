# -*- coding: utf-8 -*-
"""
This module implements community detection.
"""
from __future__ import print_function
import array
import numbers
import warnings
import networkx as nx
from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
from spatial_graphs import SpatialGraph
from .status import Status

__author__ = """Thomas Aynaud (thomas.aynaud@lip6.fr)"""
#    Copyright (C) 2009 by
#    Thomas Aynaud <thomas.aynaud@lip6.fr>
#    All rights reserved.
#    BSD license.


__PASS_MAX = 100  # NOTE: changed because of occasional infinite loops
__MIN = 0.0000001


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


# new
def decay_func(dists, ell=1., method="invpow"):
    """
    Compute the distance decay function according specified method.


    Parameters:
    ----------
    dists : numpy.ndarray
        matrix of distances between nodes
    ell : float or string, optional (default=1.)
        parameter to use in distance decay function. If string must
        be one of ['mean', 'max'] and uses mean or max of dists
    method : string, optional (default='invpow')
        distay decay function to use. If 'invpow' computes
        elementwise dists^{-ell}, if 'invexp' computes
        elementwise exp(-dists * ell)
    """
    # checks
    if (dists == 0.).any():
        raise ValueError("zero entries detected in distance matrix")
    if method not in ["invpow", "invexp"]:
        raise ValueError(f"'{method} not a valid distance decay method")
    if type(ell) == str:
        if ell in ["max", "mean"]:
            ell = 1 / getattr(np, ell)(dists[dists != np.inf])
        else:
            raise ValueError(f"invalid entry '{ell}' for ell:"
                             " must be a float or one of ['max', 'mean']")
    # actually calculate distance decay matrix
    if method == "invpow":
        return dists**(-ell), ell
    elif method == "invexp":
        return np.exp(-dists * ell), ell


# new
def inv_decay_func(x, ell, method="invpow"):
    if (x == np.infty).any():
        raise ValueError("infinite value(s) detected in x")
    if method not in ["invpow", "invexp"]:
        raise ValueError(f"'{method} not a valid distance decay method")
    if ell == 0.:
        return 1.
    else:
        if method == "invpow":
            return np.power(x, -1. / ell)
        elif method == "invexp":
            return - np.log(x) / ell


def partition_at_level(dendrogram, level):
    """Return the partition of the nodes at the given level

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1.
    The higher the level is, the bigger are the communities

    Parameters
    ----------
    dendrogram : list of dict
       a list of partitions, ie dictionaries where keys of the i+1 are the
       values of the i.
    level : int
       the level which belongs to [0..len(dendrogram)-1]

    Returns
    -------
    partition : dictionary
       A dictionary where keys are the nodes and the values are the set it
       belongs to

    Raises
    ------
    KeyError
       If the dendrogram is not well formed or the level is too high

    See Also
    --------
    best_partition : which directly combines partition_at_level and
    generate_dendrogram : to obtain the partition of highest modularity

    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendrogram = generate_dendrogram(G)
    >>> for level in range(len(dendrogram) - 1) :
    >>>     print("partition at level", level, "is", partition_at_level(dendrogram, level))  # NOQA
    """
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition


# edited
def modularity(part: dict, graph: SpatialGraph,
               weight='weight', resolution=1.,
               ell=1., decay_method="invpow",
               normalise=True):
    # checks
    if graph.is_directed():
        raise TypeError("This method is only for directed graphs.")
    if ell is None:
        raise ValueError("ell needs to be specified")

    # initialise variables
    internals = {}                            # total internal weights
    null = {}                                 # null model
    k = graph.degree(weight=weight)           # array of node degrees
    total_weight = graph.size(weight=weight)  # sum of the weights
    dists = graph.dists

    if total_weight == 0:
        raise ValueError("A graph without links has an undefined modularity")
    min_dist = min(dists[dists != 0])
    # get rid of any zero distances
    if (dists == 0).any():
        dists[dists == 0] = min_dist
    with np.errstate(divide="ignore"):
        dists_decay, _ = decay_func(dists, ell, method=decay_method)
        dists_decay[dists_decay == np.inf], _ = decay_func(
            min_dist, ell, method=decay_method)
    if normalise:
        kvec = np.array([x for _, x in k])
        kvec = kvec[:, np.newaxis]
        norm = (kvec @ kvec.T * dists_decay).sum()
        norm = total_weight / norm
    else:
        norm = 1.

    # finally, calculate internal and null sums
    for node_i, k_i in graph.degree(weight=weight):
        com = part[node_i]
        # cycle through incident nodes (upper triangle)
        for neighbor_i, datas in graph[node_i].items():
            edge_weight = datas.get(weight, 1)
            # compute total internal strength
            if part[neighbor_i] == com:
                if neighbor_i >= node_i:
                    internals[com] = internals.get(com, 0.)\
                        + float(edge_weight)
        # cycle through all nodes in graph
        for node_j, k_j in graph.degree(weight=weight):
            if part[node_j] == com:
                null[com] = null.get(com, 0.) + k_i * k_j\
                    * dists_decay[node_i, node_j]
    # sum up all communities contributions to the modularity
    Q = 0.
    for com in set(part.values()):
        Q += resolution * internals.get(com, 0.) - norm * null.get(com, 0.)
    Q /= total_weight
    # all calcs done for upper triangle so multiply by 2
    return Q


# edited
def best_partition(graph: SpatialGraph,
                   weight='weight',
                   part=None,
                   resolution=1.,
                   ell=1.,
                   decay_method="invpow",
                   randomize=None,
                   random_state=None,
                   **kwargs):
    """Compute the partition of the graph nodes which maximises the modularity.

    Wrapper-like function for generate dendogram. Removes any zero-distances.
    Parameters:
    ----------
    graph : spatial_graphs.SpatialGraph
        the SpatialGraph graph which is decomposed
    weight : string, optional (default='weight')
        the key in graph to use as weight. Default to 'weight'
    part : dict, optional (default=None)
        algorithm will initalise using this partition of the nodes. If not
        supplied the singleton partition will be used. This
        parameter can be used to limit the maximum number of communities
        found. If B communities are supplied in partition then the algorithm
        cannot return more than B communities.
    resolution : float
        controls size and number of communities found. Smaller res will find
        more, smaller  communities
    ell : float or string, optional (default=1.)
        parameter to use in distance decay function. If string must
        be one of ['mean', 'max'] and uses mean or max of dists
    method : string, optional (default='invpow')
        distay decay function to use. If 'invpow' computes
        elementwise dists^{-ell}, if 'invexp' computes
        elementwise exp(-dists * ell)
    randomize : boolean, optional
        Will randomize the node evaluation order and the community evaluation
        order to get different partitions at each call
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns:
    --------
    best_partition : dict
        final entry of dendogram containing optimal partition of the nodes
    """
    # checks
    if type(graph).__name__ != "SpatialGraph":
        raise TypeError(f"first input must be type SpatialGraph not "
                        f"{type(graph).__name__}")
    if not hasattr(graph, "dists"):
        raise AttributeError("Assign distance matrix "
                             f"to {type(graph).__name__} object")

    # process any zero entries in distance matrix
    min_dists = min(graph.dists[graph.dists != 0])
    if (graph.dists == 0).any():
        graph.dists[graph.dists == 0] = min_dists

    dendo = generate_dendrogram(graph,
                                part,
                                weight,
                                resolution,
                                ell,
                                decay_method,
                                randomize,
                                random_state)
    best_partition = partition_at_level(dendo, len(dendo) - 1)
    return best_partition


# edited
def generate_dendrogram(graph: SpatialGraph,
                        part=None,
                        weight='weight',
                        resolution=1.,
                        ell=1.,
                        decay_method="invpow",
                        randomize=None,
                        random_state=None):
    """Find communities in the graph and return the associated dendrogram
    """
    # --------------------------------------------------------------------------
    # NOTE: all code below from original package until line marked 'NEW'
    # Properly handle random state, eventually remove old `randomize` parameter
    # when `randomize` is removed, delete code up to random_state = ...
    if randomize is not None:
        warnings.warn("The `randomize` parameter will be deprecated in future "
                      "versions. Use `random_state` instead.", DeprecationWarning)
        # If shouldn't randomize, we set a fixed seed to get determinisitc results
        if randomize is False:
            random_state = 0
    # We don't know what to do if both `randomize` and `random_state` are defined
    if randomize and random_state is not None:
        raise ValueError("`randomize` and `random_state` cannot be used at the "
                         "same time")
    random_state = check_random_state(random_state)
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non-directed graph")
    # if no links use singleton partition
    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]
    # --------------------------------------------------------------------------
    # NEW
    current_graph = graph.copy()
    status = Status(method="gravity")
    def f(d): return decay_func(d, ell, decay_method)
    status.init(current_graph, weight=weight, part=part, f=f)
    status_list = list()
    _one_level(current_graph, status, weight, resolution, random_state)
    new_mod = _modularity(status, resolution)
    partition = _renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod

    current_graph = induced_graph(partition, current_graph,
                                  weight, ell, decay_method,
                                  calculate_dists=True)

    status.init(current_graph, weight=weight, f=f, norm=status.norm)

    # successive phases of Louvain until modularity decrease
    # is below __MIN threshold
    count = 0
    while True:
        # phase 1 Louvain
        _one_level(current_graph, status, weight, resolution, random_state)
        new_mod = _modularity(status, resolution)
        if new_mod - mod < __MIN:
            break
        partition = _renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod

        # phase 2 Louvain
        current_graph = induced_graph(partition, current_graph,
                                      weight, ell, decay_method,
                                      calculate_dists=True)
        status.init(current_graph, weight=weight, f=f, norm=status.norm)
        count += 1
    return status_list[:]


# edited
def induced_graph(part: dict, graph, weight="weight",
                  ell=1., decay_method="invpow", calculate_dists=True):
    """
    Calculate the graph of meta nodes and edges from the partition.


    Parameters:
    ----------
    part : dict
        partition of the original network
    graph : spatial_graphs.SpatialGraph
        original network
    weight : string, optional (default='weight')
        the key in graph to use as weight. Default to 'weight'
    ell : float, optional (default=1.)
        parameter to use in distance decay function. If string must
        be one of ['mean', 'max'] and uses mean or max of dists
    decay_method : string, optional (default='invpow')
        distay decay function to use. If 'invpow' computes
        elementwise dists^{-ell}, if 'invexp' computes
        elementwise exp(-dists * ell)
    calculate_dists : bool, optional (default=True)
        (not ready for use) whether to calculate the new meta-distances or just
        the distance decay function (reduced computation). If
        False induced_graph() returns a tuple of
        (new_graph, dists_decay)
    """
    # checks
    assert (graph.dists != 0.).all(),\
        "zero entries in distance matrix should be removed before using"\
        "induced_graph()"

    new_graph = SpatialGraph()
    new_graph.add_nodes_from(set(part.values()))
    N = max(part.values()) + 1
    new_null = np.zeros([N, N])
    old_dists = graph.dists
    old_dists_decay, ell = decay_func(old_dists, ell, decay_method)

    # get meta edge weights
    # Graph.edges() only lists (i, j) but not (j, i) for undirected
    for node_i, node_j, edge_data in graph.edges(data=True):
        eweight = edge_data.get(weight, 1)
        com_i = part[node_i]
        com_j = part[node_j]
        # NOTE: Graph.add_edge() replaces any existing edges
        meta_eweight = new_graph.get_edge_data(
            com_i, com_j, {"weight": 0}).get(weight, 1)
        new_graph.add_edge(com_i, com_j, **{weight: meta_eweight + eweight})

    new_graph.fmat = np.array(adjacency_matrix(new_graph,
                                               weight=weight).todense())

    # calculate the new null
    for node_i, k_i in graph.degree(weight=weight):
        for node_j, k_j in graph.degree(weight=weight):
            if node_j <= node_i:
                com_i = part[node_i]
                com_j = part[node_j]
                d_ij = old_dists_decay[node_i, node_j]
                p_ij = k_i * k_j * d_ij
                new_null[com_i, com_j] += p_ij
                if node_j < node_i:
                    new_null[com_j, com_i] += p_ij
    # calculate new distances
    if calculate_dists:
        new_dists = np.zeros([N, N])
        for com_i, k_i in new_graph.degree(weight="weight"):
            for com_j, k_j in new_graph.degree(weight="weight"):
                if com_j >= com_i:
                    if k_i * k_j > 0:
                        d_ij = inv_decay_func(new_null[com_i, com_j]
                                              / (k_i * k_j), ell, decay_method)
                    else:
                        # d_{ij} will be infinity for nodes with zero degree
                        d_ij = np.infty
                    new_dists[com_i, com_j] = d_ij
                    if com_j > com_i:
                        new_dists[com_j, com_i] = d_ij
        new_graph.dists = new_dists
        return new_graph
    # else only calculate distance decay matrix and return it
    else:
        dists_decay = np.zeros([N, N])
        for com_i, k_i in new_graph.degree(weight="weight"):
            for com_j, k_j in new_graph.degree(weight="weight"):
                if com_j >= com_i:
                    if k_i * k_j > 0:
                        d_ij = new_null[com_i, com_j] / (k_i * k_j)
                    dists_decay[com_i, com_j] = d_ij
                    if com_j > com_i:
                        dists_decay[com_j, com_i] = d_ij
            return new_graph, dists_decay


def _renumber(dictionary):
    """Renumber the values of the dictionary from 0 to n
    """
    values = set(dictionary.values())
    target = set(range(len(values)))

    if values == target:
        # no renumbering necessary
        ret = dictionary.copy()
    else:
        # add the values that won't be renumbered
        renumbering = dict(zip(target.intersection(values),
                               target.intersection(values)))
        # add the values that will be renumbered
        renumbering.update(dict(zip(values.difference(target),
                                    target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret


def load_binary(data):
    """Load binary graph as used by the cpp implementation of this algorithm
    """
    data = open(data, "rb")

    reader = array.array("I")
    reader.fromfile(data, 1)
    num_nodes = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_nodes)
    cum_deg = reader.tolist()
    num_links = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_links)
    links = reader.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0

    for index in range(num_nodes):
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg

    return graph


# edited
def _one_level(graph, status, weight, resolution, random_state):
    """Compute one level of communities (phase 1 of Louvain)
    """
    modified = True  # stop iterations once this is False
    nb_pass_done = 0
    cur_mod = _modularity(status, resolution)
    new_mod = cur_mod

    # continue until no nodes move in a pass or num
    # passes exceeds __PASS_MAX
    while modified:
        if nb_pass_done >= __PASS_MAX:
            print(f"Louvain phase 1 didn't converge in {nb_pass_done} "
                  "iterations. "
                  "Try again with a different random state")
            break
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        # cycle through nodes in the graph
        for node_i, k_i in __randomize(graph.degree(weight=weight),
                                       random_state):
            com_i = status.node2com[node_i]

            # NOTE: nodes with zero degree stay in their own community
            if k_i > 0:
                ki_ins = _get_ki_in(node_i, graph, status, weight)
                kj_dists = _get_kj_dists(node_i, k_i, graph, status, weight)
                # self-loops
                # ki_in_self = graph.get_edge_data(node_i, node_i,
                #                                {weight: 0.}).get(weight, 0.)
                #kj_dist_self = k_i * status.dists_decay[node_i, node_i]
                remove_cost = - resolution * ki_ins.get(com_i, 0.) + \
                    (k_i * kj_dists.get(com_i, 0.) -
                     status.null.get(com_i, 0)) * status.norm
                _remove(node_i, com_i,
                        ki_ins.get(com_i, 0.),
                        kj_dists.get(com_i, 0.),
                        status)
                best_com = com_i
                best_increase = 0.
                # calculate modularity increase for other communities
                for com_j, ki_in in __randomize(ki_ins.items(), random_state):
                    incr = remove_cost + resolution * ki_in - \
                        kj_dists.get(com_j, 0.) * k_i * status.norm
                    if incr > best_increase:
                        best_increase = incr
                        best_com = com_j
                _insert(node_i, best_com, ki_ins.get(best_com, 0.),
                        kj_dists.get(best_com, 0.),
                        status)
                if best_com != com_i:
                    modified = True
        new_mod = _modularity(status, resolution)
        if new_mod - cur_mod < __MIN:
            break


# edited, previously __neighcom()
def _get_ki_in(node, graph, status, weight):
    k_in = {}
    # get weight for self-loops (included in all incident communities)
    ki_loop = graph[node].get(node, {"weight": 0.}).get("weight", 0.)

    # loop through neighbors of a node
    for neighbor, datas in graph[node].items():
        neighborcom = status.node2com[neighbor]
        edge_weight = datas.get(weight, 1)
        if neighbor != node:
            k_in[neighborcom] = k_in.get(neighborcom, ki_loop)\
                + edge_weight
        else:
            k_in[neighborcom] = k_in.get(neighborcom, 0.) + ki_loop
    return k_in


# edited
def _get_kj_dists(node, k_i, graph, status, weight):
    ki_self = k_i * status.dists_decay[node, node]
    kj_dists = {}
    # loop through all other nodes in graph
    for node_j, k_j in graph.degree(weight=weight):
        com_j = status.node2com[node_j]
        kj_dist = k_j * status.dists_decay[node_j, node]
        kj_dists[com_j] = kj_dists.get(com_j, 0.) + \
            kj_dist

        # if node_j != node:
        #    kj_dists[com_j] = kj_dists.get(com_j, ki_self)\
        #        + 2 * kj_dist
        # else:
        #    kj_dists[com_j] = kj_dists.get(com_j, 0.) + ki_self
    return kj_dists


# edited
def _remove(node, com, ki_in, kj_dists, status):
    """ Remove node from community com and modify status"""
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.)) - 2 * ki_in
    status.node2com[node] = -1  # assign node to no community
    k_i = status.gdegrees[node]
    status.null[com] = float(status.null.get(com, 0.))\
        - status.norm * k_i * kj_dists


# edited
def _insert(node, com, ki_in, kj_dists, status):
    """ Insert node into community and modify status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.)) + 2 * ki_in
    k_i = status.gdegrees[node]
    status.null[com] = float(status.null.get(com, 0.))\
        + status.norm * k_i * kj_dists


# edited
def _modularity(status, res):
    """
    Fast compute the modularity of the partition of the graph using
    status precomputed
    """
    Q = 0.
    links = status.total_weight
    for com in set(status.node2com.values()):
        inc = status.internals.get(com, 0.)
        null = status.null.get(com, 0.)
        norm = status.norm
        Q += (res * inc - norm * null)
    return Q / (2 * links)


def __randomize(items, random_state):
    """Returns a List containing a random permutation of items"""
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items
