from networkx import Graph, DiGraph
import networkx.convert_matrix as nxmat
import numpy as np
from numpy import array
from numpy.linalg import norm
import scipy.sparse as sp


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


# TODO: make docstring
class SpatialGraph(Graph):
    """"

    """

    def __init__(self):
        super().__init__()
        self.fmat = array([])
        self.dists = array([])
        self.part = {}
        self.locs = array([])

    # tidied
    def add_distances(self, dists):
        if type(dists) != np.ndarray:
            raise TypeError("dists must be a numpy.ndarray not"
                            f"a {type(dists).__name__}")
        self.dists = dists

    # tidied
    def add_flows(self, fmat):
        if type(fmat) != np.ndarray:
            raise TypeError("dists must be a numpy.ndarray not "
                            f"a {type(fmat).__name__}")
        self.fmat = fmat

    # tidied
    def copy(self):
        # __copy__() wasn't being implemented when I tried
        new_graph = super().copy()
        new_graph.dists = self.dists.copy()
        new_graph.fmat = self.fmat.copy()
        return new_graph

    # tidied
    # TODO: test to check dims correct and dists symmetric
    @classmethod
    def from_numpy_array(cls, fmat, **kwargs):
        """
        Construct SpatialGraph using a numpy ndarray.

        Parameters:
        -----------
        fmat : numpy.ndarray
            flow matrix used to initialise the SpatialGraph
        kwargs :
            any additional attributes to assign to the
            SpatialGraph

        Note:
        -----
        From numpy array uses the networkx method
        networkx.convert_matrix.from_numpy_array(), self
        -loops of weight 1 are represented as 1s in the
        matrix but contribute 2 to the node degree.
        """
        if sp.issparse(fmat):
            fmat = fmat.todense()
        if fmat.shape[0] != fmat.shape[1]:
            raise ValueError("flow matrix must be square")

        G = nxmat.from_numpy_array(fmat, create_using=cls)
        G.add_flows(fmat)
        for key, value in kwargs.items():
            setattr(G, key, value)

        return G

    @classmethod
    def from_gravity_benchmark(cls, lamb: float, rho: float, N=20,
                               len=10., ell=1., decay_method="invpow", seed=None):
        """
        Create an undirected gravity benchmark spatial graph.

        Parameters:
        ----------
        lamb : float
            assortativity parameter; lamb < 1 creates a graph with assortative
            community structure while lamb > 1 creates a disassortative
            community structure.
        rho : float
            density of the edges and magnitude of the weights; the synthetic
            graph will have L = rho * N * (N - 1) edges where N is the number
            of nodes in the graph.
        N : int, optional (default=20)
            number of nodes in the graph
        len : float, optional (default=10.)
            length of the l x l square for node assignments
        gamma : float, optional (default=1.)
            link probabilities between nodes distance d apart will decrease
            proportional to d^{-gamma}
        seed : int, optional (default=None)
            seed to initialise the random number generator, can be set to
            an integer for reproducible graphs

        Returns:
        -------
        g : SpatialGraph
            The synthetic spatial graph

        References:
        -----------
        .. 1. Expert et al., Uncovering space-independent communities in
        spatial networks. Proceedings of the National Academy of Sciences,
        vol. 108, pp. 7663--7668(2011)

        Examples:
        ---------
        >>> from community import SpatialGraph
        >>> g = SpatialGraph.from_gravity_benchmark(0.5, 2)
        """

        L = int(rho * N * (N - 1) / 2)
        nodes = [x for x in range(N)]

        # place nodes in space and calculate distances
        rng = np.random.default_rng(seed)
        nlocs = len * rng.random((N, 2))
        dists = np.array([[norm(a - b, 2) for b in nlocs] for a in nlocs])

        # assign random binary communities
        communities = rng.integers(0, 2, N, dtype=int)
        partition = {node: community for node, community in
                     zip(nodes, communities)}

        # begin making matrix of link probabilities
        probs = np.array([[1 if i == j else lamb for j in communities]
                         for i in communities], dtype=float)

        with np.errstate(divide='ignore'):
            dists_cost, _ = decay_func(dists, ell, decay_method)
            dists_cost[dists_cost == np.infty] = 0.

        probs = probs * dists_cost
        fmat = np.zeros([N, N])

        # take entries of upper triangle of probs matrix
        upper = np.arange(N)
        mask = upper[:, None] > upper
        probs = probs[mask]
        z = probs.sum()
        probs /= z

        indices = [(i, j) for i in range(N) for j in range(i)]
        selection = rng.choice(indices, size=L, p=probs)
        for ix in selection:
            fmat[ix[0], ix[1]] += 1
        fmat += fmat.T  # symmetrise flow matrix

        g = cls.from_numpy_array(fmat, dists=dists)
        g.part = partition

        nlocs = {node: np.array(loc) for node, loc in zip(nodes, nlocs)}
        g.locs = nlocs

        return g
