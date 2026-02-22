"""
Utility functions for tree generation and recovery algorithms.
"""

import networkx as nx
import itertools
from typing import List, Set, Tuple, Dict


def is_leaf(T: nx.Graph, node: int) -> bool:
    """
    检查节点是否是树T中的叶节点。
    叶节点的度为1（对于孤立的单节点树，度为0）。
    """
    degree = T.degree(node)
    return degree <= 1


