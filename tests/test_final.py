
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


@pytest.fixture
def sample_graphs():
    fmat = np.array([[1, 1], [1, 0]])
    dmat = np.array([[1, 1], [1, 1]])
    g0 = SpatialGraph.from_numpy_array(fmat, dists=dmat)

    fmat = np.array([[1, 1], [1, 1]])
    dmat = np.array([[1, 1], [1, 1]])
    g1 = SpatialGraph.from_numpy_array(fmat, dists=dmat)

    fmat = np.array([[1, 4, 1], [4, 0, 2], [1, 2, 0]])
    dmat = np.array([[1, 1, sqrt(5)], [1, 1, 2], [sqrt(5), 2, 1]])
    g2 = SpatialGraph.from_numpy_array(fmat, dists=dmat)
    return [g0, g1, g2]


# first set of tests
@pytest.mark.parametrize("idx, test_partition", [
    (0, {0: 0, 1: 0}),
    (1, {0: 0, 1: 0}),
    (2, {0: 0, 1: 0, 2: 0})
])
def test_trivial_fast_modularity(sample_graphs, idx, test_partition):
    TOL = 1e-8
    test_graph = sample_graphs[idx]
    test_mod = modularity(test_partition, test_graph)
    status = Status(method="gravity")
    def f(d): return decay_func(d)
    status.init_gravity(test_graph, part=test_partition, f=f, normalise=True)
    test_mod = _modularity(status, res=1.)
    assert abs(test_mod) <= TOL,\
        f"fast modularity for trivial partition {test_mod} not zero"


@pytest.mark.parametrize("idx, test_partition, ell, mod_expected", [
    (0, {0: 0, 1: 1}, 0., -1. / 8),
    (1, {0: 0, 1: 1}, 0., 1. / 6),
    (2, {0: 0, 1: 1, 2: 2}, 0., (2 - (94 / 16)) / 16)
])
def test_fast_modularity(sample_graphs, idx, test_partition, ell, mod_expected):
    TOL = 1e-8
    test_graph = sample_graphs[idx]
    status = Status(method="gravity")
    def f(d): return decay_func(d, ell=ell, method="invpow")
    status.init_gravity(test_graph, part=test_partition, f=f, normalise=True)
    mod_fast = _modularity(status, res=1.)
    assert abs(mod_fast - mod_expected) <= TOL,\
        f"fast modularity ({mod_fast}) not equal to expected modularity ({mod_expected})."


@pytest.mark.parametrize("idx, test_partition, ell", [
    (0, {0: 0, 1: 1}, 0.),
    (1, {0: 0, 1: 1}, 0.),
    (2, {0: 0, 1: 1, 2: 2}, 0.)
])
def test_both_modularities_equal(sample_graphs, idx, test_partition, ell):
    test_graph = sample_graphs[idx]

    status = Status(method="gravity")
    def f(d): return decay_func(d, ell=ell, method="invpow")
    status.init_gravity(test_graph, part=test_partition, f=f, normalise=True)

    mod_fast = _modularity(status, res=1.)
    mod_slow = modularity(test_partition, test_graph,
                          resolution=1, ell=ell, normalise=True)

    assert mod_fast == mod_slow,\
        f"fast modularity {mod_fast} not equal to slow modularity {mod_slow}."


@pytest.mark.parametrize("idx, ell", [
    (0, 0.),
    (1, 0.),
    (2, 0.)
])
def test_both_partitions_equal(sample_graphs, idx, ell):
    test_graph = sample_graphs[idx]
    part_grav = community_louvain.gravity.best_partition(test_graph, ell=ell)
    part_ng = community_louvain.best_partition(test_graph)
    assert part_grav == part_ng,\
        f"gravity partition {part_grav} should equal Newman-Girvan partition "\
        f"{part_ng} when ell=0."


@pytest.mark.parametrize("idx, test_partition, ell, expected_ki_in", [
    (0, {0: 0, 1: 1}, 0., [1., 2.]),
    (1, {0: 0, 1: 1}, 0., [1., 2.]),
    (2, {0: 0, 1: 1, 2: 2}, 0., [1., 5., 2.])
])
def test_ki_in(sample_graphs, idx, test_partition, ell, expected_ki_in):
    test_graph = sample_graphs[idx]
    status = Status(method="gravity")
    def f(d): return decay_func(d, ell=ell, method="invpow")
    status.init_gravity(test_graph, part=test_partition, f=f)

    ki_in = _get_ki_in(0, test_graph, status, "weight")
    assert list(ki_in.values()) == expected_ki_in, "k_{i,in} not as expected"


@pytest.mark.parametrize("idx, test_partition, ell, expected_kj_dists", [
    (0, {0: 0, 1: 1}, 0., [3., 1.]),
    (1, {0: 0, 1: 1}, 0., [3., 3.]),
    (2, {0: 0, 1: 1, 2: 2}, 0., [7., 6., 3.])
])
def test_kj_dists(sample_graphs, idx, test_partition, ell, expected_kj_dists):
    test_graph = sample_graphs[idx]
    status = Status(method="gravity")
    def f(d): return decay_func(d, ell=ell, method="invpow")
    status.init_gravity(test_graph, part=test_partition, f=f)

    k_i = test_graph.degree(weight="weight")[0]
    kj_dists = _get_kj_dists(0, k_i, test_graph, status, "weight")
    assert list(kj_dists.values()) == expected_kj_dists,\
        "k_{j, dists} not as expected"


@pytest.mark.parametrize("idx, test_partition, ell,\
                        expected_internals, expected_null", [
    (0, {0: 0, 1: 1}, 0., [0., 0.], [0., 1.]),
    (1, {0: 0, 1: 1}, 0., [0., 2.], [0., 9.]),
    (2, {0: 0, 1: 1, 2: 2}, 0., [0., 0., 0.], [0., 36., 9.])
])
def test_remove(sample_graphs, idx, test_partition, ell,
                expected_internals, expected_null):
    test_graph = sample_graphs[idx]

    status = Status(method="gravity")
    def f(d): return decay_func(d, ell=ell, method="invpow")
    status.init_gravity(test_graph, part=test_partition, f=f, normalise=False)

    k_i = test_graph.degree(weight="weight")[0]
    ki_ins = _get_ki_in(0, test_graph, status, "weight")
    kj_dists = _get_kj_dists(0, k_i, test_graph, status, "weight")

    _remove(0, 0, ki_ins.get(0, 0.), kj_dists.get(0, 0.), status)
    assert list(status.internals.values()) == expected_internals,\
        f"removing node 0 gave unexpected internals {list(status.internals.values())}"
    assert list(status.null.values()) == expected_null,\
        f"removing node gave unexpected null {list(status.null.values())}"


@pytest.mark.parametrize("idx, test_partition, ell,\
                        expected_internals, expected_null", [
    (0, {0: 0, 1: 1}, 0., [0., 4.], [0., 16.]),
    (1, {0: 0, 1: 1}, 0., [0., 6.], [0., 36.]),
    (2, {0: 0, 1: 1, 2: 2}, 0., [0., 10., 0.], [0., 169., 9.])
])
def test_insert(sample_graphs, idx, test_partition, ell,
                expected_internals, expected_null):
    TOL = 1e-8
    test_graph = sample_graphs[idx]

    status = Status(method="gravity")
    def f(d): return decay_func(d, ell=ell, method="invpow")
    status.init_gravity(test_graph, part=test_partition, f=f, normalise=False)

    k_i = test_graph.degree(weight="weight")[0]
    ki_ins = _get_ki_in(0, test_graph, status, "weight")
    kj_dists = _get_kj_dists(0, k_i, test_graph, status, "weight")

    _remove(0, 0, ki_ins.get(0, 0.), kj_dists.get(0, 0.), status)
    _insert(0, 1, ki_ins.get(1, 0.), kj_dists.get(1, 0.), status)

    print(status.null.values())
    print(status.internals.values())

    assert list(status.internals.values()) == expected_internals,\
        "inserting node gave unexpected result"
    assert sum(x - y for x, y in zip(list(status.null.values()), expected_null)) < TOL,\
        f"inserting node gave null {list(status.null.values())} "\
        f"but expected {expected_null}"
