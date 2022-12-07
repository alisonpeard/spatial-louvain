# spatial-louvain
Experimental modification of Thomas Aynaud's package [python-louvain](https://github.com/taynaud/python-louvain) to include spatial effects by replacing the configuration null model in the Newman-Girvan modularity function with a null model that includes spatial effects using the gravity model. Part of Python for Scientific Computing special topic project for the MSc in Mathematical Modelling and Scientific Computing, University of Oxford.

## Files
* `spatial_graphs/spatial_graphs.py`: child class of Networkx's Graph class, containing extra methods and attributes for spatial graphs such as spatial locations and a distance matrix.
* `community/newman_girvan`": original modularity optimisation code using the Louvain method.
* `community/gravity`": modifications to original code where scores are based on a gravity null model rather than configuration model.

## References
1. Louvain Community Detection, Thomas Aynaud, https://github.com/taynaud/python-louvain
2. Fast unfolding of communities in large networks, Vincent D Blondel et al J. Stat. Mech. (2008) P10008, DOI: [10.1088/1742-5468/2008/10/P10008](https://iopscience.iop.org/article/10.1088/1742-5468/2008/10/P10008)
3. Paul Expert et al. “Uncovering space-independent communities in spatial net- works”. In: Proceedings of the National Academy of Sciences 108.19 (2011), pp. 7663–7668. DOI: https://doi.org/10.1073/pnas.1018962108
