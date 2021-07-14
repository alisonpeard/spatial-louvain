"""Synthetic Core-Periphery spatial and directed nets.


"Visualise the Core-Periphery synthetic networks in space and spring-block
layout.
"""

import networkx as nx  #  noqa
import matplotlib.pyplot as plt
from spatial_graphs import SpatialDiGraph

g = SpatialDiGraph.from_cpsp_benchmark(p=0.1, rho=.5, N=50)
print(g.fmat)

# cols = [p_out, cin, c_out, c_in]
cols = ['#e86256', '#c70213', '#1d5ade', '#9ad1f5']
cmap = {node: cols[int(com)] for node, com in g.part.items()}
col_arr = [cols[int(com)] for com in g.part.values()]
weights = [0.05 * g[u][v]['weight'] for u, v in g.edges()]
degs = [2 * x for x in dict(g.degree(weight="weight")).values()]

# plot spatially
pos = g.locs
fig, ax = plt.subplots()
nx.draw(g, pos, node_size=degs,  node_color=col_arr,
        edge_color='#919394', width=weights, alpha=0.85, ax=ax,
        arrowsize=7)
fig.set_facecolor('white')
ax.set_facecolor('white')
limits = plt.axis('on')
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

# plot block-spring layout
pos = nx.drawing.layout.spring_layout(g, weight="weight", iterations=50)
fig, ax = plt.subplots()
nx.draw(g, pos, node_size=degs, node_color=col_arr,
        edge_color='#919394', width=weights, alpha=0.85,
        arrowsize=7)
fig.set_facecolor('white')
ax.set_facecolor('white')

plt.show()
