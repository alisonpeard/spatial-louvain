"""Perform a parameter search of the (ρ, λ)-space."""
import numpy as np
import community as community_louvain
from spatial_graphs import SpatialGraph
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import timeit


def colorbar(mappable, label=None):
    '''adapted from https://joseph-long.com/writing/colorbars/'''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    cbar = fig.colorbar(mappable, cax=cax,
                        orientation="horizontal", label=label)
    plt.sca(last_axes)
    return cbar


def get_nmi(lamb, rho, N=20, len=20, nruns=5):
    """Calculate NMI score for gravity and Newman-Girvan."""

    nmi_grav = 0.
    nmi_ng = 0.
    time_grav = 0.
    time_ng = 0.
    mod_grav = 0.
    mod_ng = 0.

    for _ in range(nruns):
        g = SpatialGraph.from_gravity_benchmark(lamb, rho, N, len, ell=2.)
        true_partition = g.part
        t = timeit.default_timer()
        partition_grav\
            = community_louvain.gravity.best_partition(g, resolution=2.,
                                                       ell=2.,
                                                       decay_method="invpow", random_state=1)
        time_grav += timeit.default_timer() - t
        t = timeit.default_timer()
        partition_ng = community_louvain.newman_girvan.best_partition(
            g, random_state=1)
        time_ng += timeit.default_timer() - t

        nmi_grav += normalized_mutual_info_score([*true_partition.values()],
                                                 [*partition_grav.values()])
        nmi_ng += normalized_mutual_info_score([*true_partition.values()],
                                               [*partition_ng.values()])
        mod_grav += community_louvain.gravity.modularity(partition_grav, g,
                                                         resolution=2., ell=2.,
                                                         decay_method="invpow")
        mod_ng += community_louvain.modularity(partition_ng, g)

    return nmi_grav/nruns, nmi_ng/nruns, mod_grav/nruns, mod_ng/nruns,\
        time_grav/nruns, time_ng/nruns


# define parameter search variables
nruns = 20
lambs = np.linspace(2, 0, nruns)
rhos = np.logspace(-2, 2, nruns)
nmi_grav = np.zeros([nruns, nruns])
nmi_ng = np.zeros([nruns, nruns])
mod_grav = np.zeros(nruns**2)
mod_ng = np.zeros(nruns**2)
time_grav = np.zeros(nruns**2)
time_ng = np.zeros(nruns**2)


# begin parameter search
ix = 0
for i in range(nruns):
    for j in range(nruns):
        nmi_grav[i, j], nmi_ng[i, j], mod_grav[ix],\
            mod_ng[ix], time_grav[ix], time_ng[ix] = get_nmi(lambs[i], rhos[j])
        print(f"{100 * ix / (nruns ** 2)}%")
        ix += 1

# print some statistics
print(f"Gravity model:\n--------------\n"
      f"average NMI: {np.round(np.mean(nmi_grav),4)}\n"
      f"with standard deviation: {np.round(np.std(nmi_grav),4)}\n"
      f"mean modularity: {np.round(np.mean(mod_grav), 4)}\n"
      f"partition found in mean time: {np.round(np.mean(time_grav), 6)}\n")
print(f"Newman-Girvan model:\n--------------------\n"
      f"average NMI: {np.round(np.mean(nmi_ng), 4)}\n"
      f"with standard deviation: {np.round(np.std(nmi_ng),4)}\n"
      f"mean modularity: {np.round(np.mean(mod_ng), 4)}\n"
      f"partition found in mean time: {np.round(np.mean(time_ng), 6)}")

# heatmap
# params for heatmap visualisation
xticklabels = [r'$10^{-2}$', r'$10^{-1}$', r'$10^0$', r'$10^1$', r'$10^2$']
yticklabels = [0.0, 0.25, 0.5, 0.75, 1.0]
xticks = np.linspace(0, nruns-1, 5)
yticks = np.linspace(0, nruns-1, 5)[::-1]

# visualise heatmap of the NMIs
fig, ax = plt.subplots(1, 2)
im1 = ax[0].imshow(nmi_grav, clim=(0, 1))
im2 = ax[1].imshow(nmi_ng, clim=(0, 1))
ax[0].set_title("Gravity NMI")
ax[1].set_title("Newman-Girvan NMI")
ax[0].set_xlabel(r'$\rho$')
ax[0].set_ylabel(r'$\lambda$')
ax[1].set_xlabel(r'$\rho$')
ax[1].set_ylabel(r'$\lambda$')
ax[0].set(xticks=xticks, xticklabels=xticklabels)
ax[0].set(yticks=yticks, yticklabels=yticklabels)
ax[1].set(xticks=xticks, xticklabels=xticklabels)
ax[1].set(yticks=yticks, yticklabels=yticklabels)
colorbar(im1, "NMI")
colorbar(im2, "NMI")

# surface plot of the NMIs
# TODO: axes labels
fig2, ax2 = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(rhos, lambs)
surf1 = ax2[0].plot_surface(X, Y, nmi_grav, cmap=plt.cm.viridis,
                            linewidth=0, clim=(0, 1), antialiased=False)
surf2 = ax2[1].plot_surface(X, Y, nmi_ng, cmap=plt.cm.viridis,
                            linewidth=0, clim=(0, 1), antialiased=False)
ax2[0].set_title("Gravity NMI")
ax2[1].set_title("Newman-Girvan NMI")
ax2[0].set_zlim(0, 1.)
ax2[1].set_zlim(0, 1.)
ax2[0].zaxis.set_major_formatter('{x:.02f}')
ax2[1].zaxis.set_major_formatter('{x:.02f}')
ax2[0].set_xlabel(r'$\rho$')
ax2[0].set_ylabel(r'$\lambda$')
ax2[0].set_zlabel('NMI')
ax2[1].set_xlabel(r'$\rho$')
ax2[1].set_ylabel(r'$\lambda$')
ax2[1].set_zlabel('NMI')

# Add a color bar which maps values to colors.
fig2.colorbar(surf1, ax=ax2[0], shrink=.8, aspect=20, label="NMI",
              orientation="horizontal", pad=0.1)
fig2.colorbar(surf2, ax=ax2[1], shrink=.8, aspect=20, label="NMI",
              orientation="horizontal", pad=0.1)
plt.show()
