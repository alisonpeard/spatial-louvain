from spatial_graphs import SpatialGraph
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

g = SpatialGraph.from_cerina_benchmark(
    N=100, rho=1, ell=2.0, beta=0.5, epsilon=0.5, seed=0)

print(type(g))

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
