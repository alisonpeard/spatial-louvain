#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This package implements community detection.

Package name is community but refer to python-louvain on pypi
"""

# from .community_louvain_grav import (
#     partition_at_level,
#     modularity,
#     best_partition,
#     generate_dendrogram,
#     induced_graph,
#     load_binary
# )

from .status import Status
from .newman_girvan import *
import community.gravity

# to use original methods: community_louvain.function
# to use new methods: community_louvain.gravity.function

__version__ = "0.15"
__author__ = """Thomas Aynaud (thomas.aynaud@lip6.fr)"""
#    Copyright (C) 2009 by
#    Thomas Aynaud <thomas.aynaud@lip6.fr>
#    All rights reserved.
#    BSD license.
