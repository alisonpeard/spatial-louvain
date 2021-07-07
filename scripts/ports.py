import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import community as community_louvain
from spatial_graphs import SpatialGraph


# setup variables
np.random.seed(3)
datadir = os.path.join("..", "..", "data")

# load data
dmat = np.load(os.path.join(datadir, "port_dmat.npy"))
fmat = np.array(sp.load_npz(os.path.join(
    datadir, "cargo_ports.npz")).todense())
N, _ = fmat.shape

# create primitive SpatialGraph instance
g = SpatialGraph.from_numpy_array(fmat, dists=dmat)
start = time.process_time()
partition_grav = community_louvain.gravity.best_partition(
    g, res=10., gamma="mean", decay_method="invexp")
print(
    f"finished grav partition in {(time.process_time() - start)/100} seconds")
print(f"{len(set(partition_grav.values()))}")


# --------------------------------------------------------------------------------

# start = time.process_time()
# partition_ng = community_louvain.best_partition(g)
# print(f"finished ng partition in {(time.process_time() - start)/100} seconds")

# # make data frame
# part_df = pd.DataFrame.from_dict(partition_grav, orient="index")
# degs_dict = dict(G.degree())
# print(f"number of communities from gravity: {len(set(partition_grav.values()))}")
# part_df["degree"] = part_df.apply(lambda row: degs_dict[row.name], axis=1)
# part_df.to_csv(os.path.join(datadir, "grav_partition.csv"))

# # ----------------------------------------------------------------------
# # find best partition using newman-girvan null
# start = time.process_time()
# for _ in range(1):
#     print(f"{_/1}%")
#     partition_ng = community_louvain.newman_girvan.best_partition(G, resolution=0.1)
# print(f"finished newman-girvan in average {(time.process_time() - start)/100} seconds") # 0.252

# # compute modularity of this partition
# start = time.process_time()
# # print(f"classic modularity: {community_louvain.newman_girvan.modularity(partition_ng, G)}")

# # make data frame
# part_df = pd.DataFrame.from_dict(partition_ng, orient='index')
# part_df["degree"] = part_df.apply(lambda row: degs_dict[row.name], axis=1)
# part_df.to_csv(os.path.join(datadir, "newman_girvan_partition.csv"))
