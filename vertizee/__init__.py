"""
Vertizee
========
Vertizee is an object-oriented, typed, graph library for the analysis and study of graphs.
See https://<TODO> for complete documentation.
"""

# Make certain classes available to the user as direct imports from the `vertizee` namespace.
# pylint: disable=wrong-import-position
from vertizee.exception import *

from vertizee.classes.collections.bk_tree import BKTree
from vertizee.classes.collections.fibonacci_heap import FibonacciHeap
from vertizee.classes.collections.priority_queue import PriorityQueue
from vertizee.classes.collections.range_dict import RangeDict
from vertizee.classes.collections.union_find import UnionFind
from vertizee.classes.collections.vertex_dict import VertexDict

from vertizee.classes.digraph import DiGraph, MultiDiGraph
from vertizee.classes.edge import DiEdge, Edge, EdgeType
from vertizee.classes.graph import Graph, GraphBase, MultiGraph, SimpleGraph
from vertizee.classes.shortest_path import ShortestPath
from vertizee.classes.vertex import Vertex, VertexType
