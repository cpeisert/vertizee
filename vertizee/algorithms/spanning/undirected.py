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

# pylint: disable=line-too-long
r"""
===============================================
Spanning: undirected graphs (trees and forests)
===============================================

Algorithms for finding optimum :term:`spanning trees <spanning tree>` and
:term:`forests <forest>` of :term:`undirected graphs <undirected graph>`. The asymptotic running
times use the notation that for some graph :math:`G(V, E)`, the number of vertices is
:math:`n = |V|` and the number of edges is :math:`m = |E|`.

**Recommended Tutorial**: :doc:`Spanning trees, arborescences, forests, and branchings <../../tutorials/spanning_tree_arborescence>` - |image-colab-spanning|

.. |image-colab-spanning| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/cpeisert/vertizee/blob/master/docs/source/tutorials/spanning_tree_arborescence.ipynb

Function summary
================

* :func:`spanning_tree` - Iterates over a minimum (or maximum) :term:`spanning tree` of a weighted,
  :term:`undirected graph` using Kruskal's algorithm. Running time: :math:`O(m(\log{n}))`
* :func:`optimum_forest` - Iterates over the minimum (or maximum) :term:`trees <tree>` comprising a
  :term:`spanning forest` of a weighted, undirected graph. Running time: :math:`O(m(\log{n}))`
* :func:`kruskal_optimum_forest` - Iterates over the minimum (or maximum) trees comprising a
  :term:`spanning forest` of an undirected graph using Kruskal's algorithm.
  Running time: :math:`O(m(\log{n}))`
* :func:`kruskal_spanning_tree` - Iterates over a minimum (or maximum) :term:`spanning tree` of a
  weighted, undirected graph using Kruskal's algorithm. Running time: :math:`O(m(\log{n}))`
* :func:`prim_spanning_tree` - Iterates over a minimum (or maximum) :term:`spanning tree` of a
  weighted, undirected graph using Prim's algorithm. Running time: :math:`O(m(\log{n}))`
* :func:`prim_fibonacci` - Iterates over a minimum (or maximum) :term:`spanning tree` of a weighted,
  undirected graph using Prim's algorithm implemented using a :term:`Fibonacci heap`.
  Running time: :math:`O(m + n(\log{n}))`

Detailed documentation
======================
"""

from __future__ import annotations
import collections
from typing import cast, Dict, Final, Iterator, Optional, TYPE_CHECKING, Union

from vertizee import exception
from vertizee.algorithms.algo_utils.spanning_utils import get_weight_function
from vertizee.classes.data_structures.fibonacci_heap import FibonacciHeap
from vertizee.classes.data_structures.priority_queue import PriorityQueue
from vertizee.classes.data_structures.tree import Tree
from vertizee.classes.data_structures.union_find import UnionFind
from vertizee.classes.edge import Edge, MultiEdge

if TYPE_CHECKING:
    from vertizee.classes.graph import Graph, MultiGraph
    from vertizee.classes.vertex import MultiVertex, Vertex, VertexType

INFINITY: Final[float] = float("inf")


#
# TODO(cpeisert) run benchmarks to see how much slower kruskal_optimum_forest is versus
# kruskal_spanning_tree.
#
def kruskal_optimum_forest(
    graph: Union[Graph, MultiGraph], minimum: bool = True, weight: str = "Edge__weight"
) -> Iterator[Tree[Union[Vertex, MultiVertex]]]:
    r"""Iterates over the minimum (or maximum) :term:`trees <tree>` comprising an
    :term:`optimum spanning forest` of an :term:`undirected graph` using Kruskal's algorithm.

    Running time: :math:`O(m(\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    This implementation is based on MST-KRUSKAL. :cite:`2009:clrs`

    Note:
        This algorithm is only defined for *undirected* graphs. To find the optimum forest of a
        directed graph (also called an :term:`optimum branching <branching>`), see
        :func:`optimum_directed_forest
        <vertizee.algorithms.spanning.directed.optimum_directed_forest>`.

    Args:
        graph: The undirected graph to iterate.
        minimum: Optional; True to return the minimum spanning tree, or False to return the maximum
            spanning tree. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the edge ``attr``
            dictionary. The default value ("Edge__weight") uses the edge property ``weight``.

    Yields:
        Iterator[Tree[V, E]]: An iterator over the minimum (or maximum) trees. If only one tree is
        yielded prior to ``StopIteration``, then it is a spanning tree.

    See Also:
        * :func:`kruskal_spanning_tree`
        * :func:`optimum_forest`
        * :func:`spanning_tree`
        * :class:`UnionFind <vertizee.classes.data_structures.union_find.UnionFind>`
    """
    if len(graph) == 0:
        raise exception.Unfeasible("forests are undefined for empty graphs")
    if graph.is_directed():
        raise exception.GraphTypeNotSupported(
            "graph must be undirected; for directed graphs see optimum_directed_forest"
        )

    weight_function = get_weight_function(weight, minimum=minimum)
    sign = 1 if minimum else -1
    edge_weight_pairs = [(e, sign * weight_function(e)) for e in graph.edges()]
    sorted_edges = [p[0] for p in sorted(edge_weight_pairs, key=lambda pair: pair[1])]
    union_find = UnionFind(*graph.vertices())

    vertex_to_tree: Dict[Union[Vertex, MultiVertex], Tree[Union[Vertex, MultiVertex]]] = {
        v: Tree(v) for v in graph.vertices()
    }

    for edge in sorted_edges:
        if not union_find.in_same_set(edge.vertex1, edge.vertex2):
            union_find.union(edge.vertex1, edge.vertex2)
            vertex_to_tree[edge.vertex1].add_edge(edge)

    set_iter = union_find.get_sets()
    for tree_vertex_set in set_iter:
        tree = vertex_to_tree[tree_vertex_set.pop()]
        while tree_vertex_set:
            tree.merge(vertex_to_tree[tree_vertex_set.pop()])
        yield tree


def kruskal_spanning_tree(
    graph: Union[Graph, MultiGraph], minimum: bool = True, weight: str = "Edge__weight"
) -> Iterator[Union[Edge, MultiEdge]]:
    r"""Iterates over a minimum (or maximum) :term:`spanning tree` of a weighted,
    :term:`undirected graph` using Kruskal's algorithm.

    Running time: :math:`O(m(\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    Note:
        If the graph does not contain a spanning tree, for example, if the graph is disconnected,
        no error or warning will be raised. If the total number of edges yielded equals
        :math:`|V| - 1`, then there is a spanning tree, otherwise see :func:`optimum_forest`.

    Note:
        This algorithm is only defined for *undirected* graphs. To find the spanning tree of a
        directed graph, see :func:`spanning_arborescence
        <vertizee.algorithms.spanning.directed.spanning_arborescence>`.

    Note:
        This implementation is based on MST-KRUSKAL. :cite:`2009:clrs`

    Args:
        graph: The undirected graph to iterate.
        minimum: Optional; True to return the minimum spanning tree, or False to return the maximum
            spanning tree. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the edge ``attr``
            dictionary. The default value ("Edge__weight") uses the edge property ``weight``.

    Yields:
        Union[Edge, MultiEdge]: An iterator over the edges of the minimum (or maximum) spanning tree
        discovered using Kruskal's algorithm.

    Raises:
        Unfeasible: If the graph does not contain a spanning tree, an Unfeasible exception is
            raised.

    See Also:
        * :func:`optimum_forest`
        * :func:`spanning_tree`
        * :class:`UnionFind <vertizee.classes.data_structures.union_find.UnionFind>`
    """
    if len(graph) == 0:
        raise exception.Unfeasible("spanning trees are undefined for empty graphs")
    if graph.is_directed():
        raise exception.GraphTypeNotSupported(
            "graph must be undirected; for directed graphs see optimum_directed_forest"
        )

    weight_function = get_weight_function(weight, minimum=minimum)
    sign = 1 if minimum else -1
    edge_weight_pairs = [(e, sign * weight_function(e)) for e in graph.edges()]
    sorted_edges = [p[0] for p in sorted(edge_weight_pairs, key=lambda pair: pair[1])]
    union_find = UnionFind(*graph.vertices())

    for edge in sorted_edges:
        if not union_find.in_same_set(edge.vertex1, edge.vertex2):
            union_find.union(edge.vertex1, edge.vertex2)
            yield edge


def optimum_forest(
    graph: Union[Graph, MultiGraph], minimum: bool = True, weight: str = "Edge__weight"
) -> Iterator[Tree[Union[Vertex, MultiVertex]]]:
    r"""Iterates over the minimum (or maximum) :term:`trees <tree>` comprising an
    :term:`optimum spanning forest` of an :term:`undirected graph` using Kruskal's algorithm.

    Running time: :math:`O(m(\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    Note:
        This algorithm is only defined for *undirected* graphs. To find the optimum forest of a
        directed graph (also called an :term:`optimum branching <branching>`), see
        :func:`optimum_directed_forest
        <vertizee.algorithms.spanning.directed.optimum_directed_forest>`.

    Args:
        graph: The undirected graph in which to find optimum trees.
        minimum: Optional;  True to return the minimum forest, or False to return the maximum
            forest. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the edge ``attr``
            dictionary. The default value ("Edge__weight") uses the edge property ``weight``.

    Yields:
        Iterator[Tree[V, E]]: An iterator over the minimum (or maximum) trees. If only one tree is
        yielded prior to ``StopIteration``, then it is a spanning tree.

    See Also:
        * :func:`spanning_tree`
    """
    return kruskal_optimum_forest(graph, minimum=minimum, weight=weight)


def prim_spanning_tree(
    graph: Union[Graph, MultiGraph],
    root: Optional["VertexType"] = None,
    minimum: bool = True,
    weight: str = "Edge__weight",
) -> Iterator[Union[Edge, MultiEdge]]:
    r"""Iterates over a minimum (or maximum) :term:`spanning tree` of a weighted,
    :term:`undirected graph` using Prim's algorithm.

    Running time: :math:`O(m(\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    Note:
        If the graph does not contain a spanning tree, for example, if the graph is disconnected,
        no error or warning will be raised. If the total number of edges yielded equals
        :math:`|V| - 1`, then there is a spanning tree, otherwise see :func:`optimum_forest`.

    Note:
        This algorithm is only defined for *undirected* graphs. To find the spanning tree of a
        directed graph, see :func:`spanning_arborescence
        <vertizee.algorithms.spanning.directed.spanning_arborescence>`.

    Note:
        Prim's algorithm (implemented with a binary-heap-based :term:`priority queue`) has the same
        asymptotic running time as Kruskal's algorithm. However, in practice, Kruskal's algorithm
        often outperforms Prim's algorithm, since the Vertizee implementation of Kruskal's algorithm
        uses the highly-efficient :class:`UnionFind
        <vertizee.classes.data_structures.union_find.UnionFind>` data structure.

    Note:
        This implementation is based on MST-PRIM. :cite:`2009:clrs`

    Args:
        graph: The undirected graph to iterate.
        root: Optional; The root vertex of the spanning tree to be grown. If not specified, an
            arbitrary root vertex is chosen. Defaults to None.
        minimum: Optional; True to return the minimum spanning tree, or False to return the maximum
            spanning tree. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the edge ``attr``
            dictionary. The default value ("Edge__weight") uses the edge property ``weight``.

    Yields:
        Union[Edge, MultiEdge]: Edges from the minimum (or maximum) spanning tree discovered
        using Prim's algorithm.

    See Also:
        * :func:`optimum_forest`
        * :class:`Priority Queue <vertizee.classes.data_structures.priority_queue.PriorityQueue>`
        * :func:`spanning_tree`
    """
    if len(graph) == 0:
        raise exception.Unfeasible("spanning trees are undefined for empty graphs")
    if graph.is_directed():
        raise exception.GraphTypeNotSupported(
            "graph must be undirected; for directed graphs see optimum_directed_forest"
        )
    if root is not None:
        try:
            root_vertex: Union[Vertex, MultiVertex] = graph[root]
        except KeyError as error:
            raise exception.VertexNotFound(f"root vertex '{root}' not in graph") from error
    else:
        # pylint: disable=stop-iteration-return
        root_vertex = cast(Union[Vertex, MultiVertex], next(iter(graph.vertices())))

    weight_function = get_weight_function(weight, minimum=minimum)

    predecessor: Dict[
        Union[Vertex, MultiVertex], Optional[Union[Vertex, MultiVertex]]
    ] = collections.defaultdict(lambda: None)
    """A dictionary mapping a vertex to its predecessor. A predecessor is the parent vertex in the
    spanning tree. Root vertices have predecessor None."""

    priority: Dict[Union[Vertex, MultiVertex], float] = collections.defaultdict(lambda: INFINITY)
    """Dictionary mapping a vertex to its priority. Default priority is INFINITY."""

    def prim_priority_function(v: Union[Vertex, MultiVertex]) -> float:
        return priority[v]

    priority_queue: PriorityQueue[Union[Vertex, MultiVertex]] = PriorityQueue(
        prim_priority_function
    )
    for v in graph:
        priority_queue.add_or_update(v)
    priority[root_vertex] = 0
    priority_queue.add_or_update(root_vertex)

    vertices_in_tree = set()
    tree_edge: Optional[Union[Edge, MultiEdge]] = None
    sign = 1 if minimum else -1

    while priority_queue:
        u = priority_queue.pop()
        vertices_in_tree.add(u)
        if predecessor[u]:
            parent = predecessor[u]
            assert parent is not None
            adj_vertices = u.adj_vertices() - {parent}
            tree_edge = graph.get_edge(parent, u)
        else:
            adj_vertices = u.adj_vertices()

        for v in adj_vertices:
            u_v_weight = sign * weight_function(graph.get_edge(u, v))
            if v not in vertices_in_tree and u_v_weight < priority[v]:
                predecessor[v] = u
                priority[v] = u_v_weight
                priority_queue.add_or_update(v)
        if tree_edge:
            yield tree_edge


def prim_fibonacci(
    graph: Union[Graph, MultiGraph],
    root: Optional["VertexType"] = None,
    minimum: bool = True,
    weight: str = "Edge__weight",
) -> Iterator[Union[Edge, MultiEdge]]:
    r"""Iterates over a minimum (or maximum) :term:`spanning tree` of a weighted,
    :term:`undirected graph` using Prim's algorithm implemented using a :term:`Fibonacci heap`.

    Running time: :math:`O(m + n(\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    Note:
        If the graph does not contain a spanning tree, for example, if the graph is disconnected,
        no error or warning will be raised. If the total number of edges yielded equals
        :math:`|V| - 1`, then there is a spanning tree, otherwise see :func:`optimum_forest`.

    Note:
        This algorithm is only defined for *undirected* graphs. To find the spanning tree of a
        directed graph, see :func:`spanning_arborescence
        <vertizee.algorithms.spanning.directed.spanning_arborescence>`.

    Note:
        The :term:`Fibonacci-heap <Fibonacci heap>` based implementation of Prim's algorithm is
        faster than the default :term:`binary-heap <heap>` implementation, since the DECREASE-KEY
        operation, i.e. :meth:`PriorityQueue.add_or_update()
        <vertizee.classes.data_structures.priority_queue.PriorityQueue.add_or_update>`, requires
        :math:`O(\log{n})` time for binary heaps and only :math:`O(1)` amortized time for Fibonacci
        heaps.

    Note:
        This implementation is based on MST-PRIM. :cite:`2009:clrs`

    Args:
        graph: The undirected graph to iterate.
        root: Optional; The root vertex of the spanning tree to be grown. If not specified, an
            arbitrary root vertex is chosen. Defaults to None.
        minimum: Optional; True to return the minimum spanning tree, or False to return the maximum
            spanning tree. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the edge ``attr``
            dictionary. The default value ("Edge__weight") uses the edge property ``weight``.

    Yields:
        Union[Edge, MultiEdge]: Edges from the minimum (or maximum) spanning tree discovered
        using Prim's algorithm.

    See Also:
        * :func:`optimum_forest`
        * :func:`prim_spanning_tree`
        * :class:`FibonacciHeap <vertizee.classes.data_structures.fibonacci_heap.FibonacciHeap>`
        * :func:`spanning_tree`
    """
    if len(graph) == 0:
        raise exception.Unfeasible("spanning trees are undefined for empty graphs")
    if graph.is_directed():
        raise exception.GraphTypeNotSupported(
            "graph must be undirected; for directed graphs see optimum_directed_forest"
        )
    if root is not None:
        try:
            root_vertex: Union[Vertex, MultiVertex] = graph[root]
        except KeyError as error:
            raise exception.VertexNotFound(f"root vertex '{root}' not in graph") from error
    else:
        # pylint: disable=stop-iteration-return
        root_vertex = cast(Union[Vertex, MultiVertex], next(iter(graph.vertices())))

    weight_function = get_weight_function(weight, minimum=minimum)

    predecessor: Dict[
        Union[Vertex, MultiVertex], Optional[Union[Vertex, MultiVertex]]
    ] = collections.defaultdict(lambda: None)
    """A dictionary mapping a vertex to its predecessor. A predecessor is the parent vertex in the
    spanning tree. Root vertices have predecessor None."""

    priority: Dict[Union[Vertex, MultiVertex], float] = collections.defaultdict(lambda: INFINITY)
    """Dictionary mapping a vertex to its priority. Default priority is INFINITY."""

    def prim_priority_function(v: Union[Vertex, MultiVertex]) -> float:
        return priority[v]

    fib_heap: FibonacciHeap[Union[Vertex, MultiVertex]] = FibonacciHeap(prim_priority_function)
    for v in graph:
        fib_heap.insert(v)
    priority[root_vertex] = 0
    fib_heap.update_item_with_decreased_priority(root_vertex)

    vertices_in_tree = set()
    tree_edge: Optional[Union[Edge, MultiEdge]] = None
    sign = 1 if minimum else -1

    while fib_heap:
        u = fib_heap.extract_min()
        assert u is not None  #  For mypy static type checker.
        vertices_in_tree.add(u)
        if predecessor[u]:
            parent = predecessor[u]
            assert parent is not None
            adj_vertices = u.adj_vertices() - {parent}
            tree_edge = graph.get_edge(parent, u)
        else:
            adj_vertices = u.adj_vertices()

        for v in adj_vertices:
            u_v_weight = sign * weight_function(graph.get_edge(u, v))
            if v not in vertices_in_tree and u_v_weight < priority[v]:
                predecessor[v] = u
                priority[v] = u_v_weight
                fib_heap.update_item_with_decreased_priority(v)
        if tree_edge:
            yield tree_edge


def spanning_tree(
    graph: Union["Graph", "MultiGraph"], minimum: bool = True, weight: str = "Edge__weight"
) -> Iterator[Union[Edge, MultiEdge]]:
    r"""Iterates over a minimum (or maximum) :term:`spanning tree` of a weighted,
    :term:`undirected graph` using Kruskal's algorithm.

    Running time: :math:`O(m(\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    Note:
        If the graph does not contain a spanning tree, for example, if the graph is disconnected,
        no error or warning will be raised. If the total number of edges yielded equals
        :math:`|V| - 1`, then there is a spanning tree, otherwise see :func:`optimum_forest`.

    Note:
        This algorithm is only defined for *undirected* graphs. To find the spanning tree of a
        directed graph, see :func:`spanning_arborescence
        <vertizee.algorithms.spanning.directed.spanning_arborescence>`.

    Note:
        Prim's algorithm (implemented with a binary-heap-based :term:`priority queue`) has the same
        asymptotic running time as Kruskal's algorithm. However, in practice, Kruskal's algorithm,
        (which is implemented using the highly-efficient :class:`UnionFind
        <vertizee.classes.data_structures.union_find.UnionFind>` data structure), usually
        outperforms Prim.

    Note:
        This implementation is based on MST-KRUSKAL. :cite:`2009:clrs`

    Args:
        graph: The undirected graph to iterate the spanning tree.
        minimum: Optional; True to return the minimum spanning tree, or False to return
            the maximum spanning tree. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the edge ``attr``
            dictionary. The default value ("Edge__weight") uses the edge property ``weight``.

    Yields:
        Union[Edge, MultiEdge]: An iterator over the edges of the minimum (or maximum) spanning tree
        discovered using Kruskal's algorithm.

    See Also:
        * :func:`optimum_forest`
    """
    return kruskal_spanning_tree(graph, minimum, weight)
