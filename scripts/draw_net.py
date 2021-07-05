import community
import numpy as np
from numpy import sqrt
import networkx as nx
import matplotlib.pyplot as plt
from community import Status
import community as community_louvain
from spatial_graphs import SpatialGraph

# https://networkx.org/documentation/stable/reference/drawing.html
# https://www.python-graph-gallery.com/324-map-a-color-to-network-nodes
# https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#networkx.drawing.nx_pylab.draw_networkx
# https://github.com/beyondbeneath/bezier-curved-edges-networkx

# plot with colours being attribute communities #26a5b5 #2cb7c9
#  {0, 0.1, 50} give informative graphs
g = SpatialGraph.from_gravity_benchmark(
    10, 1, N=100, len=10., gamma=10., decay_method="invexp", seed=1)
cols = ['#2a57eb' if x == 1 else '#2aebc4' for x in g.partition.values()]
weights = [0.1 + 0.1 * g[u][v]['weight'] for u, v in g.edges()]
degs = [2 * x for x in dict(g.degree(weight="weight")).values()]

# plot block-spring layout
pos = nx.drawing.layout.spring_layout(g, weight="weight", iterations=50)
fig, ax = plt.subplots()
nx.draw(g, pos, node_size=degs, node_color=cols,
        edge_color='#2db7d6', width=weights, alpha=0.85)
fig.set_facecolor('white')  #  #f7f3da
ax.set_facecolor('white')
plt.show()


# plot bipartite
g = SpatialGraph.from_gravity_benchmark(0.1, 4, N=10)
cols = ['#577af2' if x == 1 else '#b5022e' for x in g.partition.values()]
weights = [0.2 * g[u][v]['weight'] for u, v in g.edges()]
degs = [2 * x for x in dict(g.degree(weight="weight")).values()]

top = []
for node, com in g.partition.items():
    if com == 0:
        top.append(node)
pos = nx.bipartite_layout(g, top)
fig, ax = plt.subplots()
nx.draw(g, pos, node_size=degs, node_color=cols,
        edge_color='#2db7d6', width=weights, alpha=0.85)
fig.set_facecolor('white')
ax.set_facecolor('white')
plt.show()


# plot in space
g = SpatialGraph.from_gravity_benchmark(0.1, 1, N=100, gamma=2.)
cols = ['#2a57eb' if x == 1 else '#2aebc4' for x in g.partition.values()]
weights = [0.1 + 0.1 * g[u][v]['weight'] for u, v in g.edges()]
degs = [2 * x for x in dict(g.degree(weight="weight")).values()]

pos = g.locations
fig, ax = plt.subplots()
nx.draw(g, pos, node_size=degs, node_color=cols,
        edge_color='#2db7d6', width=weights, alpha=0.85)
fig.set_facecolor('white')  #  #f7f3da
ax.set_facecolor('white')
plt.show()


# plot port network
dmat = np.load(os.path.join(datadir, "port_dmat.npy"))
fmat = sp.load_npz(os.path.join(datadir, "cargo_ports.npz"))
g = SpatialGraph.from_numpy_array(fmat, distances=dmat)

weights = [0.1 + 0.1 * g[u][v]['weight'] for u, v in g.edges()]
degs = [1e-6 * x for x in dict(g.degree(weight="weight")).values()]

# plot block-spring layout
pos = nx.drawing.layout.spring_layout(g, weight="weight", iterations=50)
fig, ax = plt.subplots()
nx.draw(g, pos, node_size=degs, node_color='#2a57eb',
        edge_color='#2db7d6', width=weights, alpha=0.85)
fig.set_facecolor('#f7f3da')
ax.set_facecolor('#f7f3da')
plt.show()
