"""Export matrices for a synthetic spatial graph.

Script exhibits how to create a synthetic spatial graph and export its distance
and adjacency matrices.
"""

from os.path import join  # noqa
import networkx as nx  #  noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np
from spatial_graphs import SpatialGraph

# standard for now is:
# lamb=0.5, rho=1, ell=2, seed=0

outdir = join("..", "..", "data", "matrices")
lamb = 0.5
rho = 2.
ell = 2.
seed = 0

fmat_str = f"sg_l{lamb}_r{int(rho)}_el{ell}_s{seed}_fmat".replace('.', '-')
dmat_str = f"sg_l{lamb}_r{int(rho)}_el{ell}_s{seed}_dmat".replace('.', '-')
fmat_path = join(outdir, fmat_str)
dmat_path = join(outdir, dmat_str)

g = SpatialGraph.from_gravity_benchmark(lamb, rho, ell=ell, seed=seed)
print(f"\nSpatial Benchmark Model:\n"
      f"------------------------\n"
      f"λ : {lamb}    ρ : {rho}\n"
      f"ell : {ell}   seed : {seed}\n"
      f"number of isolated nodes: {len(list(nx.isolates(g)))}"
      )
g.export_matrices_new('test')
test = np.load('test.npz', allow_pickle=True)
fmat = test['fmat']
dmat = test['dmat']
partition = test['partition'][()]


# -----------------------------------------------------------------
# OLD WAY OF EXPORTING MATRICES
# g.export_matrices(fmat_path, dmat_path)
# fmat = sp.load_npz(f"{fmat_path}.npz")
# dmat = np.load(f"{dmat_path}.npy")
# print("Finished!")

# # visualise it
# cols = ['#2a57eb' if x == 1 else '#2aebc4' for x in g.part.values()]
# weights = [0.1 + 0.1 * g[u][v]['weight'] for u, v in g.edges()]
# degs = [2 * x for x in dict(g.degree(weight="weight")).values()]

# # plot block-spring layout
# pos = nx.drawing.layout.spring_layout(g, weight="weight", iterations=50)
# fig, ax = plt.subplots()
# nx.draw(g, pos, node_size=degs, node_color=cols,
#         edge_color='#2db7d6', width=weights, alpha=0.85)
# fig.set_facecolor('white')
# ax.set_facecolor('white')
# plt.show()
