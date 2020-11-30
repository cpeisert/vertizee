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

"""Algorithms for finding optimum :term:`spanning trees <spanning tree>` and
:term:`forests <forest>` of :term:`graphs <graph>`.

Note:
    * :math:`m = |E|` (the number of :term:`edges <edge>`)
    * :math:`n = |V|` (the number of :term:`vertices <vertex>`)
    * :math:`r \\in V` is a designated root vertex of a :term:`rooted tree`

Functions:

* :func:`spanning_tree` - Iterates over a minimum (or maximum) :term:`spanning tree` of a weighted,
  :term:`undirected graph` using Kruskal's algorithm.
* :func:`optimum_forest` - Iterates over the minimum (or maximum) :term:`trees <tree>` comprising a
  :term:`spanning forest` of a weighted, undirected graph.
* :func:`optimum_directed_forest` - Iterates over the minimum (or maximum) :term:`arborescences
  <arborescence>` comprising a directed :term:`spanning forest` of a weighted,
  :term:`directed graph <digraph>`.
* :func:`kruskal_optimum_forest` - Iterates over the minimum (or maximum) trees comprising a
  :term:`spanning forest` of an undirected graph using Kruskal's algorithm.
* :func:`kruskal_traversal` - Iterates over a minimum (or maximum) :term:`spanning tree` of a
  weighted, undirected graph using Kruskal's algorithm.
* :func:`prim` - Iterates over a minimum (or maximum) :term:`spanning tree` of a weighted,
  undirected graph using Prim's algorithm.
* :func:`prim_fibonacci` - Iterates over a minimum (or maximum) :term:`spanning tree` of a weighted,
  undirected graph using Prim's algorithm implemented using a :term:`Fibonacci heap`.
"""

from __future__ import annotations
import collections
from typing import Callable, Deque, Dict, Final, Iterator, List, Optional, Set, Union

from vertizee import exception
from vertizee.classes.data_structures.fibonacci_heap import FibonacciHeap
from vertizee.classes.data_structures.priority_queue import PriorityQueue
from vertizee.classes.data_structures.tree import Tree
from vertizee.classes.data_structures.union_find import UnionFind
from vertizee.classes.graph import DiGraph, G, Graph, MultiDiGraph, MultiGraph
from vertizee.classes.edge import DiEdge, E, Edge, MultiDiEdge, MultiEdge
from vertizee.classes.vertex import DiVertex, MultiDiVertex, MultiVertex, V, Vertex


INFINITY: Final = float("inf")


class ReverseSearchState:
    """A class to save the state of which adjacent vertices (``parents``) of a vertex (``child``)
    still have not been visited in a reverse depth-first search.

    Args:
        child: The child vertex relative to ``parents`` in a reverse depth-first search tree.
        parents: An iterator over the unvisited parents of ``child`` in a search tree.
    """
    def __init__(self, child: V, parents: Iterator[V]) -> None:
        self.child = child
        self.parents = parents


def get_weight_function(weight: str = "Edge__weight", minimum: bool = True) -> Callable[[E], float]:
    """Returns a function that accepts an edge and returns the corresponding edge weight.

    If there is no edge weight, then the edge weight is assumed to be one.

    Note:
        For multigraphs, the minimum (or maximum) edge weight among the parallel edge connections
        is returned.

    Args:
        weight: Optional; The key to use to retrieve the weight from the ``Edge.attr``
            dictionary. The default value (``Edge_weight``) uses the ``Edge.weight`` property.
        minimum: Optional; For multigraphs, if True, then the minimum weight from the parallel edge
            connections is returned, otherwise the maximum weight. Defaults to True.

    Returns:
        Callable[[E], float]: A function that accepts an edge and returns the
        corresponding edge weight.
    """

    def default_weight_function(edge: E) -> float:
        if edge._parent_graph.is_multigraph():
            if minimum:
                return min(c.weight for c in edge.connections())
            return max(c.weight for c in edge.connections())
        return edge.weight

    def attr_weight_function(edge: E) -> float:
        if edge._parent_graph.is_multigraph():
            if minimum:
                return min(c.attr.get(weight, 1.0) for c in edge.connections())
            return max(c.attr.get(weight, 1.0) for c in edge.connections())
        return edge.attr.get(weight, 1.0)

    if weight == "Edge__weight":
        return default_weight_function
    return attr_weight_function


def get_total_weight_function(weight: str = "Edge__weight") -> Callable[[E], float]:
    """Returns a function that accepts an edge and returns the total weight of the edge, which
    in the case of multiedges, includes the weights of all parallel edge connections.

    If there is no edge weight, then the edge weight is assumed to be one.

    Args:
        weight: Optional; The key to use to retrieve the weight from the ``Edge.attr``
            dictionary. The default value (``Edge_weight``) uses the ``Edge.weight`` property.

    Returns:
        Callable[[E], float]: A function that accepts an edge and returns the total weight of the
        edge, including parallel connections.
    """

    def default_total_weight_function(edge: E) -> float:
        if edge._parent_graph.is_multigraph():
            return sum(c.weight for c in edge.connections())
        return edge.weight

    def attr_total_weight_function(edge: E) -> float:
        if edge._parent_graph.is_multigraph():
            return sum(c.attr.get(weight, 1.0) for c in edge.connections())
        return edge.attr.get(weight, 1.0)

    if weight == "Edge__weight":
        return default_total_weight_function
    return attr_total_weight_function


def edmonds(
    digraph: Union[DiGraph, MultiDiGraph], minimum: bool = True, weight: str = "Edge__weight"
) -> Iterator[Tree[V, E]]:
    """Iterates over the maximum (or minimum) :term:`arborescences <arborescence>` comprising a
    directed :term:`optimum spanning forest` using Edmonds' algorithm.

TODO(cpeisert): Update the running time.
    Running time: :math:`O(m + n(\\log{n})` where :math:`m = |E|` and :math:`n = |V|`

    Since a :term:`directed forest` (also called a :term:`branching`) may be comprised of
    arborescences, where each arborescence is a single vertex, the minimum directed spanning forest
    is always the set of vertices with no edges.

    However, we define an *optimum* spanning forest to be a spanning forest with the maximum number
    of edges that either has maximum or minimum weight.

    Args:
        graph: The directed graph to iterate.
        minimum: Optional;  True to return the minimum arborescences, or False to return
            the maximum arborescences. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the ``E.attr`` dictionary. The
            default value (``Edge__weight``) uses the property ``E.weight``.

    Yields:
        Iterator[Tree[V, E]]: An iterator over the minimum (or maximum) arborescences. If only one
        arborescence is yielded prior to ``StopIteration``, then it is a
        :term:`spanning arborescence`.

    See Also:
        * :func:`spanning_tree`
        * :func:`kruskal`
        * :func:`prim`
        * :func:`prim_fibonacci`
        * :class:`UnionFind <vertizee.classes.data_structures.union_find.UnionFind>`

    Note:
        This implementation is based on the treatment by Gabow, Galil, Spencer, and Tarjan in their
        paper :download:`"Efficient algorithms for finding minimum spanning trees in undirected and
        directed graphs."
        </references/Efficient_algorithms_for_finding_min_spanning_trees_GGST.pdf>` [GGST1986]_ The
        work of Gabow et al. builds upon the Chu–Liu/Edmonds' algorithm presented in the paper
        :download:`"Optimum Branchings." </references/Optimum_Branchings_Edmonds.pdf>`. [E1986]_

    References:
     .. [E1986] Jack Edmonds. :download:`"Optimum Branchings."
            </references/Optimum_Branchings_Edmonds.pdf>` Journal of Research of the National
            Bureau of Standards Section B, 71B (4):233–240, 1967.

     .. [GGST1986] Harold N. Gabow, Zvi Galil, Thomas Spencer, and Robert E. Tarjan.
            :download:`"Efficient algorithms for finding minimum spanning trees in undirected and
            directed graphs."
            </references/Efficient_algorithms_for_finding_min_spanning_trees_GGST.pdf>`
            Combinatorica 6:109-122. Springer, 1986.
    """
    if len(digraph) == 0:
        raise exception.Unfeasible("directed forests are undefined for empty graphs")
    if not digraph.is_directed():
        raise exception.GraphTypeNotSupported("graph must be directed; see spanning_tree")

    weight_function = get_weight_function(weight, minimum=minimum)
    total_weight_function = get_total_weight_function(weight)
    sign = 1 if minimum else -1

    contracted_graph: UnionFind[Union[DiVertex, MultiDiVertex]] = UnionFind()
    """The contracted graph contains disjoint sets of vertices, where each set is comprised
    of vertices that have been contracted to form a new vertex."""

    growth_path: Deque[Union[DiVertex, MultiDiVertex]] = collections.deque()
    """The "growth path" is a path of vertices formed by selecting an arbitrary vertex :math:`s`
    and then using a depth-first strategy to repeatedly choose the root (parent vertex) of the tree
    containing :math:`s`.
    """
    growth_path_set: Set[Union[DiVertex, MultiDiVertex]] = set()

    vertex_values: Dict[V, float] = dict()
    """The value assigned to each vertex is the sum of the weights of its incoming edges."""

    exit_lists: Dict[V, List[V]] = collections.defaultdict(list)
    """This dictionary maps vertices to adjacent outgoing vertices. In the paper [GGST1986]_, the
    tracking of outgoing adjacent vertices for each vertex :math:`v` is referred to as the
    *exit list* of :math:`v`.

    The edge formed by the first vertex in an exit list is designated as *active* and the remaining
    edges are considered *passive*.
    """

    incoming_passive_edges: Dict[V, Set[V]] = collections.defaultdict(set)
    """This is a mapping from vertices to sets of vertices which form incoming passive edges, where
    the passive edges are defined by the "exit lists"."""

    active_edges: Set[E] = set()

    for v in digraph.vertices():
        contracted_graph.make_set(v)
        vertex_values[v] = sum(total_weight_function(e) for e in v.incident_edges_incoming())

    # pylint: disable=stop-iteration-return
    starting_vertex = next(iter(digraph.vertices()))

    #
    # The following are steps taken from the GGST paper.
    #
    growth_path.appendleft(starting_vertex)
    growth_path_set.add(starting_vertex)
    for v in starting_vertex.adj_vertices_incoming():
        exit_lists[v].append(starting_vertex)

    while True:
        # Growth step: select lowest (or highest) weight incoming edge from growth_path, where
        # v0 = growth_path[0] and we find the lowest weight edge (u, v0).
        if minimum:
            edge = min((e for e in growth_path[0].incident_edges_incoming()), key=weight_function)
        else:
            edge = max((e for e in growth_path[0].incident_edges_incoming()), key=weight_function)

        # Case 1: u is not on the current growth path.
        if edge.tail not in growth_path_set:
            growth_path.appendleft(edge.tail)
            exit_lists[edge.tail].clear()
        else:  # Case 2: u is on the current growth path.



    parents = iter(starting_vertex.adj_vertices_incoming())
    stack: List[ReverseSearchState] = [ReverseSearchState(starting_vertex, parents)]


    while stack:
        child = stack[-1].child
        parents = stack[-1].parents

        try:
            parent = next(parents)
        except StopIteration:
            stack.pop()
            continue



#
# TODO(cpeisert) run tests to see how much slower kruskal_optimum_forest is versus kruskal_traversal
# Also, need to test kruskal_optimum_forest on graphs with isolated vertices.
#
def kruskal_optimum_forest(
    graph: Union[Graph, MultiGraph], minimum: bool = True, weight: str = "Edge__weight"
) -> Iterator[Tree[V, E]]:
    """Iterates over the minimum (or maximum) :term:`trees <tree>` comprising an
    :term:`optimum spanning forest` of an :term:`undirected graph` using Kruskal's algorithm.

    Running time: :math:`O(m(\\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    This implementation is based on MST-KRUSKAL [CLRS2009_5]_.

    Note:
        This algorithm is only defined for *undirected* graphs. To find the optimum forest of a
        directed graph (also called an :term:`optimum branching <branching>`"), see
        :func:`optimum_directed_forest`.

    Args:
        graph: The undirected graph to iterate.
        minimum: Optional; True to return the minimum spanning tree, or False to return the maximum
            spanning tree. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the ``E.attr`` dictionary. The
            default value (``Edge__weight``) uses the property ``E.weight``.

    Yields:
        Iterator[Tree[V, E]]: An iterator over the minimum (or maximum) trees. If only one tree is
        yielded prior to ``StopIteration``, then it is a spanning tree.

    See Also:
        * :func:`kruskal_traversal`
        * :func:`optimum_directed_forest`
        * :func:`optimum_forest`
        * :func:`prim`
        * :func:`prim_fibonacci`
        * :func:`spanning_tree`
        * :class:`UnionFind <vertizee.classes.data_structures.union_find.UnionFind>`

    References:
     .. [CLRS2009_5] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                     Introduction to Algorithms: Third Edition, page 631. The MIT Press, 2009.
    """
    if len(graph) == 0:
        raise exception.Unfeasible("forests are undefined for empty graphs")
    if graph.is_directed():
        raise exception.GraphTypeNotSupported(
            "graph must be undirected; see optimum_directed_forest")

    weight_function = get_weight_function(weight, minimum=minimum)
    sign = 1 if minimum else -1
    edge_weight_pairs = [(e, sign * weight_function(e)) for e in graph.edges()]
    sorted_edges = [p[0] for p in sorted(edge_weight_pairs, key=lambda pair: pair[1])]
    union_find = UnionFind(*graph.vertices())

    set_to_tree: Dict[V, Tree] = dict()

    for edge in sorted_edges:
        if not union_find.in_same_set(edge.vertex1, edge.vertex2):
            tree_v1 = set_to_tree.pop(union_find.get_set(edge.vertex1), None)
            tree_v2 = set_to_tree.pop(union_find.get_set(edge.vertex2), None)

            union_find.union(edge.vertex1, edge.vertex2)
            representative_vertex = union_find.get_set(edge.vertex1)

            if tree_v1 is not None and tree_v2 is not None:
                tree_v1.merge(tree_v2)
            elif tree_v1 is None and tree_v2 is None:
                tree_v1 = Tree(edge.vertex1)
            elif tree_v1 is None:
                tree_v1 = tree_v2
            tree_v1.add_edge(edge)
            set_to_tree[representative_vertex] = tree_v1

    tree_roots = [parent for child, parent in union_find._parents.items() if child == parent]
    for root in tree_roots:
        if root in set_to_tree:
            yield set_to_tree[root]
        else:
            yield Tree(root)


def kruskal_traversal(
    graph: Union[Graph, MultiGraph], minimum: bool = True, weight: str = "Edge__weight"
) -> Iterator[Union[Edge, MultiEdge]]:
    """Iterates over a minimum (or maximum) :term:`spanning tree` of a weighted,
    :term:`undirected graph` using Kruskal's algorithm.

    Running time: :math:`O(m(\\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    This implementation is based on MST-KRUSKAL [CLRS2009_5]_.

    Note:
        This algorithm is only defined for *undirected* graphs. To find the spanning tree of a
        directed graph, see :func:`optimum_directed_forest`.

    Note:
        If the graph does not contain a spanning tree, for example, if the graph is disconnected,
        no error or warning will be raised. See :func:`optimum_forest`.

    Args:
        graph: The undirected graph to iterate.
        minimum: Optional; True to return the minimum spanning tree, or False to return the maximum
            spanning tree. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the ``E.attr`` dictionary. The
            default value (``Edge__weight``) uses the property ``E.weight``.

    Yields:
        Union[Edge, MultiEdge]: An iterator over the edges of the minimum (or maximum) spanning tree
        discovered using Kruskal's algorithm.

    See Also:
        * :func:`kruskal_optimum_forest`
        * :func:`optimum_directed_forest`
        * :func:`optimum_forest`
        * :func:`prim`
        * :func:`prim_fibonacci`
        * :func:`spanning_tree`
        * :class:`UnionFind <vertizee.classes.data_structures.union_find.UnionFind>`
    """
    if len(graph) == 0:
        raise exception.Unfeasible("spanning trees are undefined for empty graphs")
    if graph.is_directed():
        raise exception.GraphTypeNotSupported("graph must be undirected; see optimum_directed_forest")

    weight_function = get_weight_function(weight, minimum=minimum)
    sign = 1 if minimum else -1
    edge_weight_pairs = [(e, sign * weight_function(e)) for e in graph.edges()]
    sorted_edges = [p[0] for p in sorted(edge_weight_pairs, key=lambda pair: pair[1])]
    union_find = UnionFind(*graph.vertices())

    for edge in sorted_edges:
        if not union_find.in_same_set(edge.vertex1, edge.vertex2):
            union_find.union(edge.vertex1, edge.vertex2)
            yield edge


def optimum_directed_forest(
    digraph: Union[DiGraph, MultiDiGraph], minimum: bool = True, weight: str = "Edge__weight"
) -> Iterator[Tree[V, E]]:
    """Iterates over the minimum (or maximum) :term:`arborescences <arborescence>` comprising a
    directed :term:`optimum spanning forest` (also called an :term:`optimum branching <branching>`)
    of a weighted, :term:`directed graph`.

    Running time: :math:`O(m + n(\\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    Args:
        graph: The directed graph to iterate.
        minimum: Optional;  True to return the minimum arborescences, or False to return
            the maximum arborescences. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the ``E.attr`` dictionary. The
            default value (``Edge__weight``) uses the property ``E.weight``.

    Yields:
        Iterator[Tree[V, E]]: An iterator over the minimum (or maximum) arborescences. If only one
        arborescence is yielded prior to ``StopIteration``, then it is a
        :term:`spanning arborescence`.

    See Also:
        * :func:`spanning_tree`
        * :func:`kruskal`
        * :func:`prim`
        * :func:`prim_fibonacci`
        * :class:`UnionFind <vertizee.classes.data_structures.union_find.UnionFind>`

    Note:
        This implementation is based on the treatment by Gabow, Galil, Spencer, and Tarjan in their
        paper :download:`"Efficient algorithms for finding minimum spanning trees in undirected and
        directed graphs."
        </references/Efficient_algorithms_for_finding_min_spanning_trees_GGST.pdf>` [GGST1986]_ The
        work of Gabow et al. builds upon the Chu–Liu/Edmonds' algorithm presented in the paper
        :download:`"Optimum Branchings." </references/Optimum_Branchings_Edmonds.pdf>`. [E1986]_

    References:
     .. [E1986] Jack Edmonds. :download:`"Optimum Branchings."
            </references/Optimum_Branchings_Edmonds.pdf>` Journal of Research of the National
            Bureau of Standards Section B, 71B (4):233–240, 1967.

     .. [GGST1986] Harold N. Gabow, Zvi Galil, Thomas Spencer, and Robert E. Tarjan.
            :download:`"Efficient algorithms for finding minimum spanning trees in undirected and
            directed graphs."
            </references/Efficient_algorithms_for_finding_min_spanning_trees_GGST.pdf>`
            Combinatorica 6:109-122. Springer, 1986.
    """
    if len(digraph) == 0:
        raise exception.Unfeasible("directed forests are undefined for empty graphs")
    if not digraph.is_directed():
        raise exception.GraphTypeNotSupported("graph must be directed; see spanning_tree")

    weight_function = get_weight_function(weight, minimum=minimum)
    total_weight_function = get_total_weight_function(weight)
    sign = 1 if minimum else -1

    contracted_graph: UnionFind[Union[DiVertex, MultiDiVertex]] = UnionFind()
    """The contracted graph contains disjoint sets of vertices, where each set is comprised
    of vertices that have been contracted to form a new vertex."""

    growth_path: Deque[Union[DiVertex, MultiDiVertex]] = collections.deque()
    """The "growth path" is a path of vertices formed by selecting an arbitrary vertex :math:`s`
    and then using a depth-first strategy to repeatedly choose the root (parent vertex) of the tree
    containing :math:`s`.
    """
    growth_path_set: Set[Union[DiVertex, MultiDiVertex]] = set()

    vertex_values: Dict[V, float] = dict()
    """The value assigned to each vertex is the sum of the weights of its incoming edges."""

    exit_lists: Dict[V, List[V]] = collections.defaultdict(list)
    """This dictionary maps vertices to adjacent outgoing vertices. In the paper [GGST1986]_, the
    tracking of outgoing adjacent vertices for each vertex :math:`v` is referred to as the
    *exit list* of :math:`v`.

    The edge formed by the first vertex in an exit list is designated as *active* and the remaining
    edges are considered *passive*.
    """

    incoming_passive_edges: Dict[V, Set[V]] = collections.defaultdict(set)
    """This is a mapping from vertices to sets of vertices which form incoming passive edges, where
    the passive edges are defined by the "exit lists"."""

    active_edges: Set[E] = set()

    for v in digraph.vertices():
        contracted_graph.make_set(v)
        vertex_values[v] = sum(total_weight_function(e) for e in v.incident_edges_incoming())

    # pylint: disable=stop-iteration-return
    starting_vertex = next(iter(digraph.vertices()))

    #
    # The following are steps taken from the GGST paper.
    #
    growth_path.appendleft(starting_vertex)
    growth_path_set.add(starting_vertex)
    for v in starting_vertex.adj_vertices_incoming():
        exit_lists[v].append(starting_vertex)

    while True:
        # Growth step: select lowest (or highest) weight incoming edge from growth_path, where
        # v0 = growth_path[0] and we find the lowest weight edge (u, v0).
        if minimum:
            edge = min((e for e in growth_path[0].incident_edges_incoming()), key=weight_function)
        else:
            edge = max((e for e in growth_path[0].incident_edges_incoming()), key=weight_function)

        # Case 1: u is not on the current growth path.
        if edge.tail not in growth_path_set:
            growth_path.appendleft(edge.tail)
            exit_lists[edge.tail].clear()
        else:  # Case 2: u is on the current growth path.


    # parents = iter(starting_vertex.adj_vertices_incoming())
    # stack: List[ReverseSearchState] = [ReverseSearchState(starting_vertex, parents)]


    # while stack:
    #     child = stack[-1].child
    #     parents = stack[-1].parents

    #     try:
    #         parent = next(parents)
    #     except StopIteration:
    #         stack.pop()
    #         continue


def optimum_forest(
    graph: Union[Graph, MultiGraph], minimum: bool = True, weight: str = "Edge__weight"
) -> Iterator[Tree[V, E]]:
    """Iterates over the minimum (or maximum) :term:`trees <tree>` comprising an
    :term:`optimum spanning forest` of an :term:`undirected graph` using Kruskal's algorithm.

    Running time: :math:`O(m(\\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    Args:
        graph: The undirected graph in which to find optimum trees.
        minimum: Optional;  True to return the minimum forest, or False to return the maximum
            forest. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the ``E.attr`` dictionary. The
            default value (``Edge__weight``) uses the property ``E.weight``.

    Yields:
        Iterator[Tree[V, E]]: An iterator over the minimum (or maximum) trees. If only one tree is
        yielded prior to ``StopIteration``, then it is a spanning tree.

    See Also:
        * :func:`spanning_tree`
        * :func:`kruskal`
        * :func:`prim`
        * :func:`prim_fibonacci`
        * :class:`UnionFind <vertizee.classes.data_structures.union_find.UnionFind>`
    """
    return kruskal_optimum_forest(graph, minimum=minimum, weight=weight)


def prim(
    graph: Union[Graph, MultiGraph],
    root: Optional["VertexType"] = None,
    minimum: bool = True,
    weight: str = "Edge__weight"
) -> Iterator[Union[Edge, MultiEdge]]:
    """Iterates over a minimum (or maximum) :term:`spanning tree` of a weighted,
    :term:`undirected graph` using Prim's algorithm.

    Running time: :math:`O(m(\\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    Note:
        Prim's algorithm (implemented with a binary-heap-based :term:`priority queue`) has the same
        asymptotic running time as Kruskal's algorithm. However, in practice, Kruskal's algorithm
        often outperforms Prim's algorithm, since the Vertizee implementation of Kruskal's algorithm
        uses the highly-efficient :class:`UnionFind
        <vertizee.classes.data_structures.union_find.UnionFind>` data structure.

    This algorithm is only defined for undirected graphs. To find the spanning tree of a directed
    graph, see :func:`optimum_directed_forest`.

    Note:
        This implementation is based on MST-PRIM [CLRS2009_6]_.

    Args:
        graph: The undirected graph to iterate.
        root: Optional; The root vertex of the spanning tree to be grown. If not specified, an
            arbitrary root vertex is chosen. Defaults to None.
        minimum: Optional; True to return the minimum spanning tree, or False to return the maximum
            spanning tree. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the ``E.attr`` dictionary. The
            default value (``Edge__weight``) uses the property ``E.weight``.

    Yields:
        Union[Edge, MultiEdge]: Edges from the minimum (or maximum) spanning tree discovered
        using Prim's algorithm.

    See Also:
        * :func:`kruskal_optimum_forest`
        * :func:`kruskal_traversal`
        * :func:`optimum_directed_forest`
        * :func:`optimum_forest`
        * :func:`prim`
        * :func:`prim_fibonacci`
        * :class:`Priority Queue <vertizee.classes.data_structures.priority_queue.PriorityQueue>`
        * :func:`spanning_tree`

    References:
     .. [CLRS2009_6] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                     Introduction to Algorithms: Third Edition, page 634. The MIT Press, 2009.
    """
    if len(graph) == 0:
        raise exception.Unfeasible("spanning trees are undefined for empty graphs")
    if graph.is_directed():
        raise exception.GraphTypeNotSupported(
            "graph must be undirected; see optimum_directed_forest")
    if root is not None:
        try:
            root_vertex: V = graph[root]
        except KeyError as error:
            raise exception.VertexNotFound(f"root vertex '{root}' not in graph") from error
    else:
        # pylint: disable=stop-iteration-return
        root_vertex = next(iter(graph.vertices()))

    weight_function = get_weight_function(weight, minimum=minimum)

    predecessor: Dict[V, V] = collections.defaultdict(lambda: None)
    """A dictionary mapping a vertex to its predecessor. A predecessor is the parent vertex in the
    spanning tree. Root vertices have predecessor None."""

    priority: Dict[V, float] = collections.defaultdict(lambda: INFINITY)
    """Dictionary mapping a vertex to its priority. Default priority is INFINITY."""

    def prim_priority_function(v: Union[MultiVertex, Vertex]) -> float:
        return priority[v]

    priority_queue: PriorityQueue[Vertex] = PriorityQueue(prim_priority_function)
    for v in graph:
        priority_queue.add_or_update(v)
    priority[root_vertex] = 0
    priority_queue.add_or_update(root_vertex)

    vertices_in_tree = set()
    tree_edge: Optional[Edge] = None
    sign = 1 if minimum else -1

    while priority_queue:
        u = priority_queue.pop()
        vertices_in_tree.add(u)
        if predecessor[u]:
            parent = predecessor[u]
            adj_vertices = u.adj_vertices() - {parent}
            tree_edge = graph[parent, u]
        else:
            adj_vertices = u.adj_vertices()

        for v in adj_vertices:
            u_v_weight = sign * weight_function(graph[u, v])
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
    weight: str = "Edge__weight"
) -> Iterator[Union[Edge, MultiEdge]]:
    """Iterates over a minimum (or maximum) :term:`spanning tree` of a weighted,
    :term:`undirected graph` using Prim's algorithm implemented using a :term:`Fibonacci heap`.

    Running time: :math:`O(m + n(\\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    Note:
        The Fibonacci-heap based implementation of Prim's algorithm is faster than the default
        binary-heap implementation, since the DECREASE-KEY operation, i.e.
        :meth:`PriorityQueue.add_or_update()
        <vertizee.classes.data_structures.priority_queue.PriorityQueue.add_or_update>`, requires
        :math:`O(\\log{n})` time for binary heaps and only :math:`O(1)` amortized time for Fibonacci
        heaps.

    This algorithm is only defined for undirected graphs. To find the spanning tree of a directed
    graph, see :func:`optimum_directed_forest`.

    Note:
        This implementation is based on MST-PRIM [CLRS2009_6]_.

    Args:
        graph: The undirected graph to iterate.
        root: Optional; The root vertex of the spanning tree to be grown. If not specified, an
            arbitrary root vertex is chosen. Defaults to None.
        minimum: Optional; True to return the minimum spanning tree, or False to return the maximum
            spanning tree. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the ``E.attr`` dictionary. The
            default value (``Edge__weight``) uses the property ``E.weight``.

    Yields:
        Union[Edge, MultiEdge]: Edges from the minimum (or maximum) spanning tree discovered
        using Prim's algorithm.

    See Also:
        * :func:`kruskal_optimum_forest`
        * :func:`kruskal_traversal`
        * :func:`optimum_directed_forest`
        * :func:`optimum_forest`
        * :func:`prim`
        * :func:`prim_fibonacci`
        * :class:`Priority Queue <vertizee.classes.data_structures.priority_queue.PriorityQueue>`
        * :func:`spanning_tree`
    """
    if len(graph) == 0:
        raise exception.Unfeasible("spanning trees are undefined for empty graphs")
    if graph.is_directed():
        raise exception.GraphTypeNotSupported(
            "graph must be undirected; see optimum_directed_forest")
    if root is not None:
        try:
            root_vertex: V = graph[root]
        except KeyError as error:
            raise exception.VertexNotFound(f"root vertex '{root}' not in graph") from error
    else:
        # pylint: disable=stop-iteration-return
        root_vertex = next(iter(graph.vertices()))

    weight_function = get_weight_function(weight, minimum=minimum)

    predecessor: Dict[V, V] = collections.defaultdict(lambda: None)
    """A dictionary mapping a vertex to its predecessor. A predecessor is the parent vertex in the
    spanning tree. Root vertices have predecessor None."""

    priority: Dict[V, float] = collections.defaultdict(lambda: INFINITY)
    """Dictionary mapping a vertex to its priority. Default priority is INFINITY."""

    def prim_priority_function(v: Union[MultiVertex, Vertex]) -> float:
        return priority[v]

    fib_heap: FibonacciHeap[V] = FibonacciHeap(prim_priority_function)
    for v in graph:
        fib_heap.insert(v)
    priority[root_vertex] = 0
    fib_heap.update_item_with_decreased_priority(root_vertex)

    vertices_in_tree = set()
    tree_edge: Optional[Edge] = None
    sign = 1 if minimum else -1

    while fib_heap:
        u = fib_heap.extract_min()
        assert u is not None  #  For mypy static type checker.
        vertices_in_tree.add(u)
        if predecessor[u]:
            parent = predecessor[u]
            adj_vertices = u.adj_vertices() - {parent}
            tree_edge = graph[parent, u]
        else:
            adj_vertices = u.adj_vertices()

        for v in adj_vertices:
            u_v_weight = sign * weight_function(graph[u, v])
            if v not in vertices_in_tree and u_v_weight < priority[v]:
                predecessor[v] = u
                priority[v] = u_v_weight
                fib_heap.update_item_with_decreased_priority(v)
        if tree_edge:
            yield tree_edge


def spanning_tree(
    graph: Union[Graph, MultiGraph], minimum: bool = True, weight: str = "Edge__weight"
) -> Iterator[Union[Edge, MultiEdge]]:
    """Iterates over a minimum (or maximum) :term:`spanning tree` of a weighted,
    :term:`undirected graph` using Kruskal's algorithm.

    Running time: :math:`O(m(\\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    This algorithm is only defined for *undirected* graphs. To find the spanning tree of a directed
    graph, see :func:`optimum_directed_forest`.

    Note:
        Prim's algorithm (implemented with a binary-heap-based :term:`priority queue`) has the same
        asymptotic running time as Kruskal's algorithm. However, in practice, Kruskal's algorithm,
        (which is implemented using the highly-efficient :class:`UnionFind
        <vertizee.classes.data_structures.union_find.UnionFind>` data structure), usually
        outperforms Prim.

    Note:
        This implementation is based on MST-KRUSKAL [CLRS2009_5]_.

    Args:
        graph: The undirected graph to iterate the spanning tree.
        minimum: Optional; True to return the minimum spanning tree, or False to return
            the maximum spanning tree. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the ``E.attr`` dictionary. The
            default value (``Edge__weight``) uses the property ``E.weight``.

    Yields:
        Union[Edge, MultiEdge]: An iterator over the edges of the minimum (or maximum) spanning tree
        discovered using Kruskal's algorithm.

    See Also:
        * :func:`kruskal_optimum_forest`
        * :func:`kruskal_traversal`
        * :func:`optimum_directed_forest`
        * :func:`optimum_forest`
        * :func:`prim`
        * :func:`prim_fibonacci`
        * :func:`spanning_tree`
    """
    return kruskal(graph, minimum, weight)
