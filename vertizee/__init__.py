"""
========
Vertizee
========
Vertizee is an object-oriented, typed, graph library for the analysis and study of graphs.
See https://<TODO> for complete documentation.
"""

# Make certain classes available to the user as direct imports from the `vertizee` namespace.
# pylint: disable=wrong-import-position

from vertizee.algorithms.algo_utils.path_utils import reconstruct_path, ShortestPath

from vertizee.algorithms.algo_utils.search_utils import (
    Direction,
    Label,
    SearchResults,
    VertexSearchState,
)

from vertizee.algorithms.algo_utils.spanning_utils import (
    Cycle,
    PseudoEdge,
    PseudoGraph,
    PseudoVertex,
)

from vertizee.algorithms.connectivity.components import (
    Component,
    connected_components,
    strongly_connected_components,
    weakly_connected_components,
)

from vertizee.algorithms.paths.all_pairs import (
    all_pairs_shortest_paths,
    floyd_warshall,
    johnson,
    johnson_fibonacci,
)
from vertizee.algorithms.paths.single_source import (
    bellman_ford,
    breadth_first_search_shortest_paths,
    dijkstra,
    dijkstra_fibonacci,
    shortest_paths,
)

from vertizee.algorithms.search.breadth_first_search import (
    bfs,
    bfs_vertex_traversal,
    bfs_labeled_edge_traversal,
)

from vertizee.algorithms.search.depth_first_search import (
    dfs,
    dfs_labeled_edge_traversal,
    dfs_preorder_traversal,
    dfs_postorder_traversal,
)

from vertizee.algorithms.spanning.undirected import (
    kruskal_optimum_forest,
    kruskal_spanning_tree,
    optimum_forest,
    prim_spanning_tree,
    prim_fibonacci,
    spanning_tree,
)

from vertizee.classes.collection_views import ItemsView, ListView, SetView
from vertizee.classes.edge import (
    Attributes,
    create_edge_label,
    DiEdge,
    Edge,
    EdgeBase,
    EdgeConnectionData,
    EdgeConnectionView,
    EdgeType,
    MultiDiEdge,
    MultiEdge,
    MultiEdgeBase,
    MutableEdgeBase,
)
from vertizee.classes.graph import DiGraph, G, Graph, GraphBase, MultiDiGraph, MultiGraph
from vertizee.classes.primitives_parsing import GraphPrimitive
from vertizee.classes.vertex import (
    DiVertex,
    MultiDiVertex,
    MultiVertex,
    V,
    V_co,
    Vertex,
    VertexBase,
    VertexLabel,
    VertexType,
)
from vertizee.classes.data_structures.fibonacci_heap import FibonacciHeap
from vertizee.classes.data_structures.priority_queue import PriorityQueue
from vertizee.classes.data_structures.tree import Tree
from vertizee.classes.data_structures.union_find import UnionFind
from vertizee.classes.data_structures.vertex_dict import VertexDict

from vertizee.exception import (
    AlgorithmError,
    EdgeNotFound,
    GraphTypeNotSupported,
    NegativeWeightCycle,
    NoPath,
    ParallelEdgesNotAllowed,
    SelfLoopsNotAllowed,
    Unfeasible,
    VertexNotFound,
    VertizeeException,
    VertizeeError,
)

from vertizee.io.adj_list import read_adj_list, read_weighted_adj_list, write_adj_list_to_file
