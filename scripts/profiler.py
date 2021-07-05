# https://medium.com/fintechexplained/advanced-python-learn-how-to-profile-python-code-1068055460f9
# https://medium.com/fintechexplained/time-complexities-of-python-data-structures-ddb7503790ef
# https://stackoverflow.com/questions/3927628/how-can-i-profile-python-code-line-by-line
# https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html

import community
import numpy as np
from numpy import sqrt
from community import Status
import community as community_louvain
from community.gravity import best_partition, induced_graph
from spatial_graphs import SpatialGraph

import cProfile
import functools
import pstats
import tempfile
import timeit


def time_me(number_of_times):
    def decorator(func):
        @functools.wraps(func)
        def wraps(*args, **kwargs):
            r = timeit.timeit(func, number=number_of_times)
            print(r/number_of_times)
        return wraps
    return decorator

def profile_me(func):
    @functools.wraps(func)
    def wraps(*args, **kwargs):
        file = tempfile.mktemp()
        profiler = cProfile.Profile()
        profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(file)
        metrics = pstats.Stats(file)
        metrics.strip_dirs().sort_stats('time').print_stats(100)
    return wraps

#@time_me(100)
@profile_me
def get_best_partition(graph, res):
    best_partition(graph, res=res)

@profile_me
def get_induced_graph(partition, graph):
    induced_graph(partition, graph)

g = SpatialGraph.from_gravity_benchmark(0.5, 1., N=20)
get_best_partition(g, res=1.)
#get_induced_graph(partition, g)


# 3rd June
# profile me: _one_level and get_kj_dists take the most time


# line-by-line in iPython
# %load_ext line_profiler
# %lprun -f induced_graph induced_graph(partition, g)
# %lprun? for more info
# RESULTS (6 digits+ first)
#--------------------------------------------------------------
# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#    330     10000     823877.0     82.4     36.3              if (node1, node2) not in evaluated:
#    342     10000     790970.0     79.1     34.9              if (com1, com2) not in evaluated:
#    326     10000     159637.0     16.0      7.0              k_i = old_degs[node1]
#    327     10000     159031.0     15.9      7.0              k_j = old_degs[node2]
#    343      5050      79658.0     15.8      3.5                  k_alp = new_degs[com1]
#    344      5050      78364.0     15.5      3.5                  k_bet = new_degs[com2]
#    349      5050      35173.0      7.0      1.6                  new_dists[com1, com2] = inv_decay_func(P[com1, com2] / (k_alp * k_bet), gamma, decay_method)
#    352      4950      29501.0      6.0      1.3                      new_dists[com2, com1] = inv_decay_func(P[com1, com2] / (k_alp * k_bet), gamma, decay_method)
# maybe just make non-repeating list for this and cycle through?
# or better way
# RESULTS
#--------------------------------------------------------------
# get_best_partition profile (top entries):
#   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#         1    0.000    0.000    2.224    2.224 cprofiler.py:29(get_best_partition)
#         1    0.000    0.000    2.224    2.224 gravity.py:182(best_partition)
#         1    0.000    0.000    2.222    2.222 gravity.py:215(generate_dendrogram)
#        1    1.621    1.621    2.154    2.154 gravity.py:286(induced_graph)
#
# get_induced_graph profile (top cumtime entries):
#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#        1    0.000    0.000    2.154    2.154 cprofiler.py:33(get_induced_graph)
#        1    1.616    1.616    2.153    2.153 gravity.py:286(induced_graph)
#     30100    0.021    0.000    0.475    0.000 reportviews.py:445(__getitem__)
#     30100    0.106    0.000    0.452    0.000 {built-in method builtins.sum}
#   1586270    0.253    0.000    0.346    0.000 reportviews.py:450(<genexpr>)
#   1566511    0.095    0.000    0.095    0.000 {method 'get' of 'dict' objects}
#     10000    0.020    0.000    0.042    0.000 gravity.py:70(inv_decay_func)
#     10000    0.007    0.000    0.021    0.000 {method 'all' of 'numpy.generic' objects}
#     10002    0.002    0.000    0.014    0.000 _methods.py:59(_all)
#     10006    0.012    0.000    0.012    0.000 {method 'reduce' of 'numpy.ufunc' objects}
#      2585    0.005    0.000    0.006    0.000 graph.py:820(add_edge)
#      5172    0.004    0.000    0.005    0.000 reportviews.py:772(__iter__)
#      2586    0.002    0.000    0.004    0.000 convert_matrix.py:888(<genexpr>)

# NOTE:
# inv_decay function gets called 1000 times can I do anything about this