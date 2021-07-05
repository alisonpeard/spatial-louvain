"""Test the spatia_graphs.spatial_graphs module."""
from _pytest.fixtures import fixture
import pytest

try:
    import numpy as np
    from numpy import sqrt
    from spatial_graphs import SpatialGraph
    from community.gravity import modularity, decay_func, inv_decay_func, best_partition
    from community.gravity import _modularity, _get_ki_in, _get_kj_dists, _remove,\
        _insert
    import community as community_louvain
    from community import Status
except ImportError:
    pass


def test_imports():
    import numpy as np
    from numpy import sqrt
    from spatial_graphs import SpatialGraph
    from community.gravity import modularity, decay_func, inv_decay_func
    from community.gravity import _modularity, _get_ki_in, _get_kj_dists,\
        _remove, _insert
    import community as community_louvain
    from community import Status


@pytest.mark.parametrize("test_fmat", [
    np.array([[1, 4, 1], [4, 0, 2]]),
    np.array([[0, 1]]),
    np.array([[1, 0]])
])
def test_fmat_square(test_fmat):
    with pytest.raises(ValueError):
        SpatialGraph.from_numpy_array(test_fmat)


@pytest.mark.parametrize("test_fmat, test_dmat", [
    (np.array([[1, 4, 1], [4, 0, 2], [1, 2, 0]]),
     np.array([[1, 1, sqrt(5)], [1, 1, 2], [sqrt(5), 2, 1]])),
    (np.array([[0, 1], [1, 1]]), np.array([[1, 1], [1, 1]]))
])
def test_dmat(test_fmat, test_dmat):
    test_graph = SpatialGraph.from_numpy_array(test_fmat, dists=test_dmat)
    assert (test_graph.dists == test_dmat).all(),\
        "dmat not preserved as expected"


# test from_numpy_array method
@pytest.mark.parametrize("test_fmat", [
    np.array([[1, 4, 1], [4, 0, 2], [1, 2, 0]]),
    np.array([[0, 1], [1, 1]]),
    np.array([[1, 0], [0, 1]])
])
def test_fmat(test_fmat):
    test_graph = SpatialGraph.from_numpy_array(test_fmat)
    assert (test_graph.fmat == test_fmat).all(),\
        "fmat not preserved as expected"


# test from_gravity_benchmark method
@pytest.mark.parametrize("N, seed, fmat, dmat", [
    (2, 0, np.array([[0., 1.], [1., 0.]]),
     np.array([[0., 6.47566253], [6.47566253, 0.]])),
    (3, 0, np.array([[0., 0., 1.], [0., 0., 2.], [1., 2., 0.]]),
     np.array([[0., 6.47566253, 6.66703581],
              [6.47566253, 0., 11.8307512],
              [6.66703581, 11.8307512, 0.]]))
])
def test_reproducible(N, seed, fmat, dmat):
    TOL = 1e-8
    test_graph = SpatialGraph.from_gravity_benchmark(0.5, 1, N=N, seed=seed)
    print(test_graph.dists)
    print(dmat)
    assert (test_graph.fmat == fmat).all(), "fmat not as expected"
    assert np.linalg.norm(test_graph.dists - dmat, 2) < TOL,\
        f"dmat not as expected {(test_graph.dists == dmat)}"
