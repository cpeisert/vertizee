"""
========
Vertizee
========
Vertizee is an object-oriented, typed, graph library for the analysis and study of graphs.
See https://<TODO> for complete documentation.
"""

# Make certain classes available to the user as direct imports from the `vertizee` namespace.
# pylint: disable=wrong-import-position
from vertizee.exception import *

from vertizee.classes.data_structures.fibonacci_heap import FibonacciHeap
from vertizee.classes.data_structures.priority_queue import PriorityQueue
from vertizee.classes.data_structures.union_find import UnionFind
from vertizee.classes.data_structures.vertex_dict import VertexDict

from vertizee.classes.digraph import DiGraph, MultiDiGraph
from vertizee.classes.edge import DiEdge, Edge, EdgeType
from vertizee.classes.graph import Graph, GraphBase, MultiGraph, SimpleGraph
from vertizee.classes.vertex import Vertex, VertexType
