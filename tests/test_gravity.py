"""For testing the gravity-louvain module. Only put passing tests in here"""

from _pytest.fixtures import fixture
import pytest

try:
    import numpy as np
    from numpy.linalg import norm
    from numpy import sqrt
    from networkx.linalg.graphmatrix import adjacency_matrix
    from spatial_graphs import SpatialGraph
    import community as community_louvain
    from community.gravity import induced_graph, modularity
except ImportError:
    pass


def test_imports():
    import numpy as np
    from numpy.linalg import norm
    from numpy import sqrt
    import networkx as nx
    from networkx.linalg.graphmatrix import adjacency_matrix
    from spatial_graphs import SpatialGraph
    from community.gravity import induced_graph, best_partition, modularity
#-----------------------------------------------------------------------------------------
# useful functions
def get_grav_null(g, weight="weight", ell=1.):
    dists = g.dists
    k = np.array([x for _, x in g.degree(weight=weight)])
    k = k[:, np.newaxis]
    with np.errstate(divide="ignore"):
        dists_decay = dists**(-ell)
        dists_decay[dists_decay == np.inf] = 0.
    null = (k @ k.T) * dists_decay
    return null


#-----------------------------------------------------------------------------------------
# test agglomeration step
@pytest.fixture
def sample_graphs():
    fmat = np.array([[1, 4, 1], [4, 0, 2], [1, 2, 0]])
    dmat = np.array([[1, 1, sqrt(5)], [1, 1, 2], [sqrt(5), 2, 1]])
    g1 = SpatialGraph.from_numpy_array(fmat, dists=dmat)

    fmat = np.array([[1, 1], [1, 1]])
    dmat = np.array([[1, 1], [1, 1]])
    g2 = SpatialGraph.from_numpy_array(fmat, dists=dmat)

    fmat = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    dmat = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    g3 = SpatialGraph.from_numpy_array(fmat, dists=dmat)
    return [g1, g2, g3]


@pytest.mark.parametrize("idx", [0])
def test_induced_graph(sample_graphs, idx):
    test_graph = sample_graphs[idx]
    assert type(test_graph).__name__ == "SpatialGraph", "incorrect induced graph type"


@pytest.mark.parametrize("idx, test_partition", [(0, {0: 0, 1: 0, 2: 0}), (1, {0: 0, 1: 0})])
def test_trivial_modularity(sample_graphs, idx, test_partition):
    TOL = 1e-12
    test_graph = sample_graphs[idx]
    test_mod = modularity(test_partition, test_graph)
    assert abs(test_mod) < TOL,\
        f"nonzer modularity {test_mod} for trivial partition"


@ pytest.mark.parametrize("idx, test_partition, expected_degs", [
    (0, {0: 0, 1: 0, 2: 1}, [(0, 13), (1, 3)])
])
def test_degs(sample_graphs, idx, test_partition, expected_degs):
    test_graph = sample_graphs[idx]
    g_new = induced_graph(test_partition, test_graph, ell=1.)
    degs = g_new.degree(weight="weight")
    assert list(degs) == expected_degs, "induced degrees not as expected"


@pytest.mark.parametrize("idx, test_partition", [
    (0, {0: 0, 1: 0, 2: 1})
])
def test_induced_size(sample_graphs, idx, test_partition):
    test_graph = sample_graphs[idx]
    m = test_graph.size(weight="weight")
    g_new = induced_graph(test_partition, test_graph, ell=1.)
    m_new = g_new.size(weight="weight")
    assert m == m_new, "graph sizes not preserved for agglomeration step"


# TODO: choose just one
@pytest.mark.parametrize("idx, test_partition, expected_fmat", [
    (0, {0: 0, 1: 0, 2: 1}, np.array([[5., 3.], [3., 0.]]))
])
def test_induced_flowmat(sample_graphs, idx, test_partition, expected_fmat):
    test_graph = sample_graphs[idx]
    g_new = induced_graph(test_partition, test_graph)
    fmat = np.array(adjacency_matrix(g_new, weight="weight").todense())
    assert (fmat == expected_fmat).all(), "induced flow matrix not as expected"

# TODO: choose just one
@pytest.mark.parametrize("idx, test_partition, expected_null, TOL", [
    (0, {0: 0, 1: 0, 2: 1}, np.array([[169., 18.39148551], [18.39148551, 9.]]), 1e-6)
])
def test_induced_null(sample_graphs, idx, test_partition, expected_null, TOL):
    test_graph = sample_graphs[idx]
    g_new = induced_graph(test_partition, test_graph, ell=1.)
    null = get_grav_null(g_new)
    assert norm(null - expected_null) < TOL, f"induced null matrix not as expected"


@pytest.mark.parametrize("idx, test_partition, expected_dists, TOL", [
    (0, {0: 0, 1: 0, 2: 1}, np.array([[1., 2.12054649],
     [2.12054649, 1.]]), 1e-6)
])
def test_induced_dists(sample_graphs, idx, test_partition,
                       expected_dists, TOL):
    test_graph = sample_graphs[idx]
    g_new = induced_graph(test_partition, test_graph, ell=1.)
    dmat = g_new.dists
    assert norm(dmat - expected_dists) < TOL,\
        "induced distances not as expected"


@pytest.mark.parametrize("idx, test_partition", [
    (0, {0: 0, 1: 0, 2: 1})
])
def test_induced_modularity(sample_graphs, idx, test_partition):
    test_graph = sample_graphs[idx]
    g_new = induced_graph(test_partition, test_graph, ell=1.)
    part_new = {x: x for x in range(g_new.size())}  # singleton partition
    mod = modularity(test_partition, test_graph, ell=1.)
    mod_new = modularity(part_new, g_new, ell=1.)
    assert mod == mod_new, "modularities not preserved for agglomeration step"


@pytest.mark.parametrize("idx, test_partition", [
    (0, {0: 0, 1: 0, 2: 1}),
    (1, {0: 0, 1: 0})
])
def test_ell_zero(sample_graphs, idx, test_partition):
    TOL = 1e-6
    test_graph = sample_graphs[idx]
    mod_grav = modularity(test_partition, test_graph, ell=0.)
    mod_ng = community_louvain.modularity(test_partition, test_graph)
    assert abs(mod_grav - mod_ng) < TOL,\
        f"{mod_grav} doesn't equal {mod_ng}. Newman-Girvan "\
        "and gravity modularities should be the same for ell=0."
