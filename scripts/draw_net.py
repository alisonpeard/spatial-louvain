"""Scripts for plotting synthetic spatial graphs."""
import community  #  noqa
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from spatial_graphs import SpatialGraph


#  {0, 0.1, 50} give visually informative graphs
g = SpatialGraph.from_gravity_benchmark(
    10, 1, N=100, len=10., ell=10., decay_method="invexp", seed=1)
cols = ['#2a57eb' if x == 1 else '#2aebc4' for x in g.part.values()]
weights = [0.1 + 0.1 * g[u][v]['weight'] for u, v in g.edges()]
degs = [2 * x for x in dict(g.degree(weight="weight")).values()]

# plot block-spring layout
pos = nx.drawing.layout.spring_layout(g, weight="weight", iterations=50)
fig, ax = plt.subplots()
nx.draw(g, pos, node_size=degs, node_color=cols,
        edge_color='#2db7d6', width=weights, alpha=0.85)
fig.set_facecolor('white')
ax.set_facecolor('white')


# plot in spatial location
g = SpatialGraph.from_gravity_benchmark(0.1, 1, N=100, ell=2.)
cols = ['#2a57eb' if x == 1 else '#2aebc4' for x in g.part.values()]
weights = [0.1 + 0.1 * g[u][v]['weight'] for u, v in g.edges()]
degs = [2 * x for x in dict(g.degree(weight="weight")).values()]

pos = g.locs
fig, ax = plt.subplots()
nx.draw(g, pos, node_size=degs, node_color=cols,
        edge_color='#2db7d6', width=weights, alpha=0.85)
fig.set_facecolor('white')
ax.set_facecolor('white')
plt.show()
