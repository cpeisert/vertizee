# Copyright 2020 The Vertizee Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Algorithms for finding minimum/maximum spanning trees of graphs."""

from __future__ import annotations
from typing import Iterator, Optional, TYPE_CHECKING

from vertizee.classes.data_structures.fibonacci_heap import FibonacciHeap
from vertizee.classes.data_structures.priority_queue import PriorityQueue
from vertizee.classes.data_structures.union_find import UnionFind
from vertizee.classes.vertex import Vertex
from vertizee.exception import GraphTypeNotSupported, VertexNotFound

if TYPE_CHECKING:
    from vertizee.classes.graph_base import GraphBase
    from vertizee.classes.edge import DiEdge, EdgeType
    from vertizee.classes.vertex import VertexType

INFINITY = float("inf")


def _weight_function(edge: "EdgeType", weight: str = "Edge__weight", minimum: bool = True) -> float:
    """Returns the weight of a given edge.

    If there is no edge weight, then the edge weight is assumed to be one.  If ``graph`` is a
    multigraph, the minimum (or maximum) edge weight over all parallel edges is returned.

    Args:
        edge: The edge whose weight is returned.
        weight: Optional; The key to use to retrieve the weight from the ``Edge.attr``
            dictionary. The default value (``Edge_weight``) uses the ``Edge.weight`` property.
        minimum: Optional; True to return the minimum edge weight or False to return the
            maximum edge weight.

    Returns:
        float: The edge weight.
    """
    if weight == "Edge__weight":
        edge_weight = edge.weight
    else:
        edge_weight = edge.attr.get(weight, 1)

    if len(edge.parallel_edge_weights) > 0:
        if minimum:
            min_parallel = min(edge.parallel_edge_weights)
            edge_weight = min(edge_weight, min_parallel)
        else:
            max_parallel = max(edge.parallel_edge_weights)
            edge_weight = max(edge_weight, max_parallel)
    if minimum:
        return edge_weight
    else:
        return -1 * edge_weight


def spanning_arborescence_ggst(
    graph: "GraphBase", weight: str = "Edge__weight", minimum: bool = True
) -> Iterator["DiEdge"]:
    """Iterates over a minimum (or maximum) spanning arborescence of a weighted, directed graph.

    An arborescence is defined to be a directed spanning tree of a given digraph.

    Args:
        graph: The directed graph to iterate.
        weight: Optional; [description]. Defaults to "Edge__weight".
        minimum: Optional;  True to return the minimum spanning arborescence, or False to return
            the maximum spanning arborescence. Defaults to True.

    Yields:
        Iterator[DiEdge]: An iterator over the directed edges of the minimum (or maximum) spanning
        arborescence.
    """
    # TODO(cpeisert) - Implement


def spanning_tree_kruskal(
    graph: "GraphBase", weight: str = "Edge__weight", minimum: bool = True
) -> Iterator["EdgeType"]:
    """Iterates over a minimum (or maximum) spanning tree of a weighted, undirected graph using
    Kruskal's algorithm.

    Running time: :math:`O(E \\log V)`

    This algorithm is only defined for *undirected* graphs. To find the spanning tree of a directed
    graph, see :func:`spanning_arborescence_ggst`.

    The :class:`Edge <vertizee.classes.edge.Edge>` class has a built-in ``weight`` property, which
    is used by default to determine edge weights (a.k.a. edge lengths). Alternatively, a key name
    may be provided to lookup the weight in the ``Edge.attr`` dictionary. If ``graph`` is a
    multigraph, the minimum (or maximum) edge weight over all parallel edges is returned.

    Note:
        This implementation is based on MST-KRUSKAL [CLRS2009_5]_.

    Args:
        graph: The undirected graph to iterate.
        weight: Optional; The key to use to retrieve the weight from the ``Edge.attr``
            dictionary. The default value (``Edge_weight``) uses the ``Edge.weight`` property.
        minimum: Optional; True to return the minimum spanning tree, or False to return
            the maximum spanning tree. Defaults to True.

    Returns:
        Iterator[EdgeType]: An iterator over the edges of the minimum (or maximum) spanning tree
        discovered using Kruskal's algorithm.

    See Also:
        * :class:`Edge <vertizee.classes.edge.Edge>`
        * :func:`spanning_arborescence_ggst`
        * :func:`spanning_tree_prim`
        * :func:`spanning_tree_prim_fibonacci`
        * :class:`UnionFind <vertizee.classes.data_structures.union_find.UnionFind>`

    References:
     .. [CLRS2009_5] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                     Introduction to Algorithms: Third Edition, page 631. The MIT Press, 2009.
    """
    if graph.is_directed_graph():
        raise GraphTypeNotSupported("graph must be undirected; see spanning_arborescence_ggst")

    edge_weight_pairs = [(e, _weight_function(e, weight, minimum)) for e in graph.edges]
    sorted_edges = [p[0] for p in sorted(edge_weight_pairs, key=lambda pair: pair[1])]
    union_find = UnionFind(*graph.vertices)

    for edge in sorted_edges:
        if not union_find.in_same_set(edge.vertex1, edge.vertex2):
            union_find.union(edge.vertex1, edge.vertex2)
            yield edge


def spanning_tree_prim(
    graph: "GraphBase",
    root: Optional["VertexType"] = None,
    weight: str = "Edge__weight",
    minimum: bool = True,
) -> Iterator["EdgeType"]:
    """Iterates over a minimum (or maximum) spanning tree of a weighted, undirected graph using
    Prim's algorithm.

    Running time: :math:`O(E \\log V)`

    Note:
        Prim's algorithm (implemented with a binary-heap-based priority queue) has the same
        asymptotic running time as Kruskal's algorithm. However, in practice, Kruskal's algorithm
        often outperforms Prim's algorithm, since the Vertizee implementation of Kruskal's algorithm
        uses the highly-efficient :class:`UnionFind
        <vertizee.classes.data_structures.union_find.UnionFind>` data structure.

    This algorithm is only defined for undirected graphs. To find the spanning tree of a directed
    graph, see :func:`spanning_arborescence_ggst`.

    The :class:`Edge <vertizee.classes.edge.Edge>` class has a built-in ``weight`` property, which
    is used by default to determine edge weights (a.k.a. edge lengths). Alternatively, a key name
    may be provided to lookup the weight in the ``Edge.attr`` dictionary. If ``graph`` is a
    multigraph, the minimum (or maximum) edge weight over all parallel edges is returned.

    Note:
        This implementation is based on MST-PRIM [CLRS2009_6]_.

    Args:
        graph: The undirected graph to iterate.
        root: Optional; The root vertex of the minimum spanning tree to be grown. If not
            specified, an arbitrary root vertex is chosen. Defaults to None.
        weight: Optional; The key to use to retrieve the weight from the ``Edge.attr``
            dictionary. The default value (``Edge_weight``) uses the ``Edge.weight`` property.
        minimum: Optional; True to return the minimum spanning tree, or False to return
            the maximum spanning tree. Defaults to True.

    Returns:
        Iterator[EdgeType]: An iterator over the edges of the minimum (or maximum) spanning tree
        discovered using Prim's algorithm.

    See Also:
        * :class:`Edge <vertizee.classes.edge.Edge>`
        * :class:`Priority Queue <vertizee.classes.data_structures.priority_queue.PriorityQueue>`
        * :func:`spanning_arborescence_ggst`
        * :func:`spanning_tree_kruskal`
        * :func:`spanning_tree_prim_fibonacci`

    References:
     .. [CLRS2009_6] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                     Introduction to Algorithms: Third Edition, page 634. The MIT Press, 2009.
    """
    PRIM_PARENT_KEY = "__prim_parent"
    PRIM_PRIORITY_KEY = "__prim_priority"

    if graph.is_directed_graph():
        raise GraphTypeNotSupported("graph must be undirected; see spanning_arborescence_ggst")
    if root is not None:
        r: Vertex = graph[root]
        if r is None:
            raise VertexNotFound("root vertex not found in the graph")
    else:
        if graph.vertex_count > 0:
            r = next(iter(graph.vertices))
        else:
            return iter([])

    def prim_priority_function(v: Vertex) -> float:
        return v.attr[PRIM_PRIORITY_KEY]

    priority_queue: PriorityQueue[Vertex] = PriorityQueue(prim_priority_function)
    for v in graph:
        v.attr[PRIM_PARENT_KEY] = None
        v.attr[PRIM_PRIORITY_KEY] = INFINITY
        priority_queue.add_or_update(v)
    r.attr[PRIM_PRIORITY_KEY] = 0
    priority_queue.add_or_update(r)

    vertices_in_tree = set()
    tree_edge: EdgeType = None

    while len(priority_queue) > 0:
        u = priority_queue.pop()
        vertices_in_tree.add(u)
        if u.attr[PRIM_PARENT_KEY] is not None:
            parent = u.attr[PRIM_PARENT_KEY]
            adj_vertices = u.adj_vertices - {parent}
            tree_edge = graph[parent, u]
        else:
            adj_vertices = u.adj_vertices

        for v in adj_vertices:
            u_v_weight = _weight_function(graph[u, v], weight, minimum)
            if v not in vertices_in_tree and u_v_weight < v.attr[PRIM_PRIORITY_KEY]:
                v.attr[PRIM_PARENT_KEY] = u
                v.attr[PRIM_PRIORITY_KEY] = u_v_weight
                priority_queue.add_or_update(v)
        if tree_edge:
            yield tree_edge


def spanning_tree_prim_fibonacci(
    graph: "GraphBase", root: Vertex = None, weight: str = "Edge__weight", minimum: bool = True
) -> Iterator["EdgeType"]:
    """Iterates over a minimum (or maximum) spanning tree of a weighted, undirected graph using
    Prim's algorithm implemented using a Fibonacci heap.

    Running time: :math:`O(E + V \\log V)`

    This Fibonacci-heap based implementation of Prim's algorithm is faster than the default
    binary-heap implementation, since the DECREASE-KEY operation, i.e.
    :meth:`PriorityQueue.add_or_update()
    <vertizee.classes.data_structures.priority_queue.PriorityQueue.add_or_update>`, requires
    :math:`O(\\log V)` time for binary heaps and only :math:`O(1)` amortized time for Fibonacci
    heaps.

    This algorithm is only defined for undirected graphs. To find the spanning tree of a directed
    graph, see :func:`spanning_arborescence_ggst`.

    The :class:`Edge <vertizee.classes.edge.Edge>` class has a built-in ``weight`` property, which
    is used by default to determine edge weights (a.k.a. edge lengths). Alternatively, a key name
    may be provided to lookup the weight in the ``Edge.attr`` dictionary. If ``graph`` is a
    multigraph, the minimum (or maximum) edge weight over all parallel edges is returned.

    Note:
        This implementation is based on MST-PRIM [CLRS2009_6]_.

    Args:
        graph: The undirected graph to iterate.
        root: Optional; The root vertex of the minimum spanning tree to be grown. If not
            specified, an arbitrary root vertex is chosen. Defaults to None.
        weight: Optional; The key to use to retrieve the weight from the ``Edge.attr``
            dictionary. The default value (``Edge_weight``) uses the ``Edge.weight`` property.
        minimum: Optional; True to return the minimum spanning tree, or False to return
            the maximum spanning tree. Defaults to True.

    Returns:
        Iterator[EdgeType]: An iterator over the edges of the minimum (or maximum) spanning tree
        discovered using Prim's algorithm.

    See Also:
        * :class:`Edge <vertizee.classes.edge.Edge>`
        * :class:`Priority Queue <vertizee.classes.data_structures.priority_queue.PriorityQueue>`
        * :func:`spanning_arborescence_ggst`
        * :func:`spanning_tree_kruskal`
        * :func:`spanning_tree_prim`
    """
    PRIM_PARENT_KEY = "__prim_parent"
    PRIM_PRIORITY_KEY = "__prim_priority"

    if graph.is_directed_graph():
        raise GraphTypeNotSupported("graph must be undirected; see spanning_arborescence_ggst")
    if root is not None:
        r: Vertex = graph[root]
        if r is None:
            raise VertexNotFound("root vertex not found in the graph")
    else:
        if len(graph.vertices) > 0:
            r = next(iter(graph.vertices))
        else:
            return iter([])

    def prim_priority_function(v: Vertex) -> float:
        return v.attr[PRIM_PRIORITY_KEY]

    fib_heap: FibonacciHeap[Vertex] = FibonacciHeap(prim_priority_function)
    for v in graph:
        v.attr[PRIM_PARENT_KEY] = None
        v.attr[PRIM_PRIORITY_KEY] = INFINITY
        fib_heap.insert(v)
    r.attr[PRIM_PRIORITY_KEY] = 0
    fib_heap.update_item_with_decreased_priority(r)

    vertices_in_tree = set()
    tree_edge: EdgeType = None

    while len(fib_heap) > 0:
        u = fib_heap.extract_min()
        vertices_in_tree.add(u)
        if u.attr[PRIM_PARENT_KEY] is not None:
            parent = u.attr[PRIM_PARENT_KEY]
            adj_vertices = u.adj_vertices - {parent}
            tree_edge = graph[parent, u]
        else:
            adj_vertices = u.adj_vertices

        for v in adj_vertices:
            u_v_weight = _weight_function(graph[u, v], weight, minimum)
            if v not in vertices_in_tree and u_v_weight < v.attr[PRIM_PRIORITY_KEY]:
                v.attr[PRIM_PARENT_KEY] = u
                v.attr[PRIM_PRIORITY_KEY] = u_v_weight
                fib_heap.update_item_with_decreased_priority(v)
        if tree_edge:
            yield tree_edge
