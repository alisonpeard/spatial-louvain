from spatial_graphs import SpatialGraph
from community.gravity import _one_level, decay_func
from community.status import Status
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

N = 100
ell = 2.


def f(d): return decay_func(d, ell, "invpow")


g = SpatialGraph.from_cerina_benchmark(
    N=N, rho=1, ell=ell, beta=0.5, epsilon=0.5, seed=0)
part = {node: node for node in range(N)}  # singleton partition to start
dmin = min(g.dists[g.dists != 0])
g.dists[g.dists == 0] = dmin


status = Status(method="gravity")
status.init(g, weight="weight", part=part, f=f)
status_list = list()
_one_level(g, status, "weight", 1.)


cols = ['#2a57eb' if x == 1 else '#2aebc4' for x in g.part.values()]
weights = [0.1 + 0.1 * g[u][v]['weight'] for u, v in g.edges()]
degs = [2 * x for x in dict(g.degree(weight="weight")).values()]

pos = g.locs
fig, ax = plt.subplots()
nx.draw(g, pos, node_size=degs, node_color=cols,
        edge_color='#2db7d6', width=weights, alpha=0.85, ax=ax)
fig.set_facecolor('white')
ax.set_facecolor('white')
limits = plt.axis('on')
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)


plt.show()
