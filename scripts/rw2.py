
import numpy as np
from numpy import sqrt
from spatial_graphs import SpatialGraph
from community.gravity import modularity, decay_func, inv_decay_func, best_partition
from community.gravity import _modularity, _get_ki_in, _get_kj_dists, _remove,\
    _insert
import community as community_louvain
from spatial_graphs import SpatialGraph


test_graph = SpatialGraph.from_gravity_benchmark(0.1, 1, ell=2., seed=1)
test_part = test_graph.part
part = community_louvain.best_partition(test_graph, random_state=0)
print(part)
