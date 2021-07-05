"""
Modified community_status.

For use with functions in community.gravity and community.newman_girvan modules.
"""

# coding=utf-8
import numpy as np


class Status(object):
    """
    To handle several data in one struct.

    Could be replaced by named tuple, but don't want to depend on python 2.6
    """
    node2com = {}     # community assignments
    total_weight = 0  # total sum of edge weights
    internals = {}    # sum of each community's internal weights
    degrees = {}      # total degree sum for each community
    gdegrees = {}     # each node's degree
    loops = {}        # sum of self-loop weights
    null = {}         # null model for expectations (gravity only)
    norm = None       # normalisation constant for null model (gravity only)

# edited
    def __init__(self, method="newman_girvan"):
        self.method = method
        self.node2com = {}
        self.total_weight = 0
        self.degrees = {}
        self.gdegrees = {}
        self.internals = {}
        self.loops = {}
        self.dists = {}        # matrix of distances between nodes
        self.dists_decay = {}  # matrix of distance decays
        self.null = {}

    def __str__(self):
        return ("node2com : " + str(self.node2com) + " degrees : "
                + str(self.degrees) + " internals : " + str(self.internals)
                + " total_weight : " + str(self.total_weight))

# edited
    def copy(self):
        """Perform a deep copy of status"""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight.copy()
        new_status.loops = self.loops.copy()
        new_status.dists = self.dists.copy()
        new_status.dists_decay = self.dists_decay.copy()
        new_status.null = self.null.copy()

# edited
    def init(self, *args, **kwargs):
        """Initialize the status of a graph depending on method."""
        getattr(self, f"init_{self.method}")(*args, **kwargs)

    def init_newman_girvan(self, graph, weight, part=None):
        """Initialize the status of a graph with every node in one community"""
        count = 0
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.total_weight = graph.size(weight=weight)
        if part is None:
            for node in graph.nodes():
                self.node2com[node] = count
                deg = float(graph.degree(node, weight=weight))
                if deg < 0:
                    error = "Bad node degree ({})".format(deg)
                    raise ValueError(error)
                self.degrees[count] = deg
                self.gdegrees[node] = deg
                edge_data = graph.get_edge_data(
                    node, node, default={weight: 0})
                self.loops[node] = float(edge_data.get(weight, 1))
                self.internals[count] = self.loops[node]
                count += 1
        else:
            for node in graph.nodes():
                com = part[node]
                self.node2com[node] = com
                deg = float(graph.degree(node, weight=weight))
                self.degrees[com] = self.degrees.get(com, 0) + deg
                self.gdegrees[node] = deg
                inc = 0.
                for neighbor, datas in graph[node].items():
                    edge_weight = datas.get(weight, 1)
                    if edge_weight <= 0:
                        error = "Bad graph type ({})".format(type(graph))
                        raise ValueError(error)
                    if part[neighbor] == com:
                        if neighbor == node:
                            inc += float(edge_weight)
                        else:
                            inc += float(edge_weight) / 2.
                self.internals[com] = self.internals.get(com, 0) + inc

    # new
    def init_gravity(self, graph, weight="weight", part=None,
                     f=None, dists_decay=None,
                     normalise=True, norm=None):
        """
        Initialise the status of a spatial graph with supplied partition.


        Parameters:
        -----------
        graph : spatial_graphs.SpatialGraph
            the SpatialGraph graph which is decomposed
        weight : string, optional (default='weight')
            the key in graph to use as weight. Default to 'weight'
        part : dict, optional (default=None)
            partition the positive and negative sums of modularity
            are calculated according to. If None then the singleton
            partition is used
        f : function, optional (default=None)
            distance decay function, if None then a precomputed
            matrix of distance costs must be passed
        dists_decay : numpy.ndarray, optional (default=None)
            precomputed matrix of distance costs, if None then
            a distance decay function must be passed
        normalise : bool, optional (default=True)
            whether to normalise the null model so that total
            flow is the same as that in the graph's flow matrix
        norm : float, optional (default=None)
            normalisation constant, if None supplied this is
            calculated from the graph
        """
        self.dists_decay = {}  # matrix of distance decays
        self.node2com = {}     # node community assignments
        self.null = {}         # null model for expected flows

        self.total_weight = graph.size(weight=weight)

        # calculate dists_decay matrix if not supplied
        if dists_decay is None:
            if not graph.dists.shape[0] > 0:
                raise AttributeError("SpatialGraph must have a nonempty "
                                     "distance matrix attribute if no "
                                     "distance decay matrix supplied")
            if not f:
                raise ValueError("Supply a decay function if not supplying a "
                                 "distance decay matrix")
            if (graph.dists == 0.).any():
                raise ValueError("zero entries in distance matrix should be removed "
                                 "before initalising Status")
            dists_decay, _ = f(graph.dists)
            self.dists_decay = dists_decay
        else:
            self.dists_decay = dists_decay

        # use singleton partition if no partition specified
        if part is None:
            for node_i, k_i in graph.degree(weight=weight):
                if k_i < 0:
                    raise ValueError(f"Bad node degree ({k_i}).")
                self.node2com[node_i] = node_i
                self.degrees[node_i] = k_i
                self.gdegrees[node_i] = k_i
                edge_data = graph.get_edge_data(node_i,
                                                node_i, default={weight: 0})
                self.loops[node_i] = float(edge_data.get(weight, 1))
                self.internals[node_i] = self.loops[node_i]
                self.null[node_i] = k_i * k_i * dists_decay[node_i, node_i]
        # use specific partition if specified
        else:
            # calculate all degrees first
            for node_i, k_i in graph.degree(weight=weight):
                if k_i < 0:
                    raise ValueError(f"Bad node degree ({k_i}).")
                com = part[node_i]
                self.node2com[node_i] = com
                self.degrees[com] = self.degrees.get(com, 0) + k_i
                self.gdegrees[node_i] = k_i

            # calculate null and internal weight sums
            for node_i, k_i in graph.degree(weight=weight):
                internals = 0.  # sums of internal weights for com
                null = 0.       # sum of null model entries for com
                com = part[node_i]

                # get internal weight sums using neighbours
                for node_j, datas in graph[node_i].items():
                    edge_weight = datas.get(weight, 1)
                    if edge_weight <= 0:
                        raise ValueError(
                            f"Bad graph type ({type(graph)}).")
                    if part[node_j] == com:
                        if node_i != node_j:
                            internals += edge_weight
                        else:
                            internals += 2 * edge_weight

                # populate null matrix P using all other nodes
                for node_j, k_j in graph.degree(weight=weight):
                    if part[node_j] == com:
                        null += k_i * k_j * dists_decay[node_i, node_j]
                self.internals[com] = self.internals.get(com, 0.) + internals
                self.null[com] = self.null.get(com, 0.) + null

        # normalisation constant
        if normalise and not self.norm:
            k = np.array([x for x in self.gdegrees.values()])
            k = k[:, np.newaxis]
            norm = (k @ k.T * dists_decay).sum()
            norm = self.total_weight / norm
            self.norm = 2 * norm
        if not normalise:
            self.norm = 1.
        # TODO: fix
        # finally, normalise the null model
        # self.null = {com: self.norm * null for com, null in self.null.items()}
