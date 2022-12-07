# spatial-louvain
Experimental modification of Thomas Aynaud's package [python-louvain](https://github.com/taynaud/python-louvain) to include spatial effects by replacing the configuration null model in the Newman-Girvan modularity function with a null model that includes spatial effects using the gravity model. 

## Files
* `spatial_graphs/spatial_graphs.py`: child class of Networkx's Graph class, containing extra methods and attributes for spatial graphs such as spatial locations and a distance matrix.
* `community/newman_girvan`": original modularity optimisation code using the Louvain method.
* `community/gravity`": modifications to original code where scores are based on a gravity null model rather than configuration model.

## References
1. Louvain Community Detection, Thomas Aynaud, https://github.com/taynaud/python-louvain
2. Fast unfolding of communities in large networks, Vincent D Blondel et al J. Stat. Mech. (2008) P10008, DOI: 10.1088/1742-5468/2008/10/P10008
