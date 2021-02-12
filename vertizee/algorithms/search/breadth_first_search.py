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
"""
============================
Search: breadth-first search
============================

Algorithms for breadth-first search. The asymptotic running times use the
notation that for some graph :math:`G(V, E)`, the number of vertices is :math:`n = |V|` and the
number of edges is :math:`m = |E|`.

**Recommended Tutorial**: :doc:`Search <../../tutorials/search>` - |image-colab-search|

.. |image-colab-search| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/cpeisert/vertizee/blob/master/docs/source/tutorials/search.ipynb

Function summary
================

* :func:`bfs` - Performs a breadth-first-search and provides detailed results (e.g. a :term:`forest`
  of breadth-first-search :term:`trees <tree>` and :term:`edge` classification). Running time:
  :math:`O(m + n)`
* :func:`bfs_labeled_edge_traversal` - Iterates over the labeled :term:`edges <edge>` of a
  breadth-first search traversal. Running time: :math:`O(m + n)`
* :func:`bfs_vertex_traversal` - Iterates over the :term:`vertices <vertex>` using a breadth-first
  search. Running time: :math:`O(m + n)`

Detailed documentation
======================
"""

from __future__ import annotations
import collections
from typing import cast, Deque, Dict, Final, Iterator, Optional, Set, Tuple
from typing import TYPE_CHECKING, Union, ValuesView

from vertizee import exception
from vertizee.algorithms.algo_utils.search_utils import (
    Direction,
    get_adjacent_to_child,
    Label,
    SearchResults,
    VertexSearchState,
)
from vertizee.classes import edge as edge_module
from vertizee.classes.data_structures.tree import Tree
from vertizee.classes.data_structures.union_find import UnionFind
from vertizee.classes.vertex import V, V_co

if TYPE_CHECKING:
    from vertizee.classes.edge import E_co
    from vertizee.classes.graph import GraphBase
    from vertizee.classes.vertex import VertexType

INFINITY: Final[float] = float("inf")


def bfs(
    graph: GraphBase[V_co, E_co], source: Optional[VertexType] = None, reverse_graph: bool = False
) -> SearchResults[V_co, E_co]:
    """Performs a breadth-first-search and provides detailed results (e.g. a :term:`forest` of
    breadth-first-search :term:`trees <tree>` and edge classification).

    Running time: :math:`O(m + n)`

    Note:
        If a ``source`` is not specified, then vertices are repeatedly selected until all
        :term:`components <connected component>` in the graph have been searched.

    Note:
        Breadth-first search does not support :term:`cycle` detection. For cycle detection, use
        :func:`dfs <vertizee.algorithms.search.depth_first_search.dfs>` (depth-first search).

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to begin the search. When ``source`` is
            specified, only the component reachable from the source is searched. Defaults to None.
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal as if
            the graph were reversed (i.e. the :term:`reverse graph <reverse>`). Defaults to False.

    Returns:
        SearchResults: The results of the breadth-first search.

    #### TODO(cpeisert): Update example below

    Example:
        >>> import vertizee as vz
        >>> from vertizee.algorithms import search
        >>> g = vz.Graph()
        >>> g.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 5), (6, 7)])
        >>> results = search.breadth_first_search(g)
        >>> results.vertices_preorder
        [3, 1, 0, 2, 5, 4, 7, 6]
        >>> [str(edge) for edge in results.edges_in_discovery_order]
        ['(1, 3)', '(0, 1)', '(1, 2)', '(3, 5)', '(4, 5)', '(6, 7)']

    See Also:
        * :class:`SearchResults <vertizee.algorithms.algo_utils.search_utils.SearchResults>`
        * :class:`Tree <vertizee.classes.data_structures.tree.Tree>`

    Note:
        The references for this algorithm are documented in :func:`bfs_labeled_edge_traversal`.
    """
    results = SearchResults(graph, depth_first_search=False)

    labeled_edge_tuple_iterator = bfs_labeled_edge_traversal(
        graph, source=source, reverse_graph=reverse_graph
    )

    bfs_tree: Optional[Tree[V_co, E_co]] = None
    for (parent, child, label, direction, _) in labeled_edge_tuple_iterator:
        vertex = child

        if direction == Direction.PREORDER:
            if label == Label.TREE_ROOT:
                bfs_tree = Tree(root=vertex)
                results._search_tree_forest.append(bfs_tree)
                results._vertices_preorder.append(vertex)
        elif direction == Direction.POSTORDER:
            results._vertices_postorder.append(vertex)

        if label == Label.TREE_EDGE and direction == Direction.PREORDER:
            edge = graph.get_edge(parent, child)
            results._tree_edges.add(edge)
            assert bfs_tree is not None
            bfs_tree.add_edge(edge)
            results._edges_in_discovery_order.append(edge)
            results._vertices_preorder.append(vertex)
        elif label == Label.BACK_EDGE:
            results._back_edges.add(graph.get_edge(parent, child))
        elif label == Label.CROSS_EDGE:
            results._cross_edges.add(graph.get_edge(parent, child))
        elif label == Label.FORWARD_EDGE:
            results._forward_edges.add(graph.get_edge(parent, child))

    return results


def bfs_labeled_edge_traversal(
    graph: GraphBase[V_co, E_co],
    source: Optional[VertexType] = None,
    depth_limit: Optional[float] = None,
    reverse_graph: bool = False,
) -> Iterator[Tuple[V_co, V_co, str, str, float]]:
    """Iterates over the labeled :term:`edges <edge>` of a breadth-first search traversal.

    Running time: :math:`O(m + n)`

    Note:
        If ``source`` is specified, then the traversal only includes the graph
        :term:`component <connected component>` containing the ``source`` vertex.

    For :term:`directed graphs <digraph>`, setting ``reverse_graph`` to True will generate
    vertices as if the graph were :term:`reversed <reverse>`.

    Args:
        graph: The graph to search.
        source: The source vertex from which to discover reachable vertices.
        depth_limit: Optional; The depth limit of the search. Defaults to None (no limit).
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the :term:`reverse graph <reverse>`). Defaults
            to False.

    Yields:
        Tuple[Vertex, Vertex, str, str, int]: An iterator over tuples of the form
        ``(parent, child, label, search_direction, depth)`` where ``(parent, child)`` is the edge
        being explored in the breadth-first search.

        The ``label`` is one of the strings:

            1. "tree_root" - :math:`(u, u)`, where :math:`u` is the root vertex of a BFS tree.
            2. "tree_edge" - edge :math:`(u, v)` is a tree edge if :math:`v` was first discovered by
               exploring edge :math:`(u, v)`.
            3. "back_edge" - back edge :math:`(u, v)` connects vertex :math:`u` to ancestor
               :math:`v` in a breadth-first tree. Per *Introduction to Algorithms*, self loops are
               considered back edges. :cite:`2009:clrs`
            4. "forward_edge" - non-tree edges :math:`(u, v)` connecting a vertex :math:`u` to a
               descendant :math:`v` in a breadth-first tree.
            5. "cross_edge" - All other edges, which may go between vertices in the same
               breadth-first tree as long as one vertex is not an ancestor of the other, or they go
               between vertices in different breadth-first trees.

        The ``search_direction`` is the direction of traversal and is one of the strings:

            1. "preorder" - the traversal discovered new vertex `child` in the BFS.
            2. "postorder" - the traversal finished visiting vertex `child` in the BFS.
            3. "already_discovered" - the traversal found a non-tree edge connecting to a vertex
               that was already discovered.

        The ``depth`` is the count of edges between ``child`` and the root vertex in its
        breadth-first search tree. If  the edge :math:`(parent, child)` is not a tree edge (or the
        tree root), the ``depth`` defaults to infinity.

    Example:
        The labels reveal the complete transcript of the breadth-first search algorithm.

        >>> from pprint import pprint
        >>> import vertizee as vz
        >>> from vertizee.algorithms import search
        >>> g = vz.DiGraph([(0, 1), (1, 2), (2, 1)])
        >>> pprint(list(search.bfs_labeled_edge_traversal(g, source=0)))
        [(0, 0, 'tree_root', 'preorder', 0),
         (0, 1, 'tree_edge', 'preorder', 1),
         (0, 0, 'tree_root', 'postorder', 0),
         (1, 2, 'tree_edge', 'preorder', 2),
         (0, 1, 'tree_edge', 'postorder', 1),
         (2, 1, 'back_edge', 'already_discovered', inf),
         (1, 2, 'tree_edge', 'postorder', 2)]

    See Also:
        * :class:`Direction <vertizee.algorithms.algo_utils.search_utils.Direction>`
        * :class:`Label <vertizee.algorithms.algo_utils.search_utils.Label>`
        * :class:`SearchResults <vertizee.algorithms.algo_utils.search_utils.SearchResults>`

    Note:
        This function uses ideas from the NetworkX function:
        `networkx.algorithms.traversal.breadth_first_search.generic_bfs_edges
        <https://github.com/networkx/networkx/blob/master/networkx/algorithms/traversal/breadth_first_search.py>`_
        :cite:`2008:hss`

        The NetworkX function was in turn adapted from David Eppstein's breadth-first search
        function in `PADS`. :cite:`2015:eppstein`

        The edge labeling of this function is based on the treatment in *Introduction to
        Algorithms*. :cite:`2009:clrs`

        The feature to allow depth limits is based on Korf. :cite:`1985:korf`
    """
    if len(graph) == 0:
        raise exception.Unfeasible("search is undefined for an empty graph")

    classified_edges: Set[str] = set()
    """The set of edges that have been classified so far by the breadth-first search into one of:
    tree edge, back edge, cross edge, or forward edge."""

    predecessor: Dict[V_co, Optional[V_co]] = collections.defaultdict(lambda: None)
    """The predecessor is the parent vertex in the BFS tree. Root vertices have predecessor None.
    In addition, if a source vertex is specified, unreachable vertices also have predecessor None.
    """

    search_trees: UnionFind[V_co] = UnionFind()
    """UnionFind data structure, where each disjoint set contains the vertices from a breadth-first
    search tree."""

    seen: Set[V_co] = set()
    """A set of the vertices discovered so far during a breadth-first search."""

    vertex_depth: Dict[V_co, float] = collections.defaultdict(lambda: INFINITY)

    vertices: Union[ValuesView[V_co], Set[V_co]]
    if source is None:
        vertices = graph.vertices()
    else:
        vertices = {graph[source]}
    if depth_limit is None:
        depth_limit = INFINITY

    for vertex in vertices:
        if vertex in seen:
            continue

        depth_now = 0
        search_trees.make_set(vertex)  # New BFS tree root.
        seen.add(vertex)
        vertex_depth[vertex] = depth_now

        children = get_adjacent_to_child(child=vertex, parent=None, reverse_graph=reverse_graph)
        queue: Deque[VertexSearchState[V_co]] = collections.deque()
        queue.append(VertexSearchState(vertex, children, depth_now))

        yield vertex, vertex, Label.TREE_ROOT, Direction.PREORDER, depth_now

        # Explore the bread-first search tree rooted at `vertex`.
        while queue:
            parent_state: VertexSearchState[V_co] = queue.popleft()
            parent = parent_state.parent

            for child in parent_state.children:
                edge_label = edge_module.create_edge_label(parent, child, graph.is_directed())

                if child not in seen:  # Discovered new vertex?
                    seen.add(child)
                    if parent_state.depth is not None:
                        depth_now = parent_state.depth + 1
                    vertex_depth[child] = depth_now
                    predecessor[child] = parent
                    search_trees.make_set(child)
                    search_trees.union(parent, child)
                    classified_edges.add(edge_label)

                    yield parent, child, Label.TREE_EDGE, Direction.PREORDER, depth_now

                    grandchildren = get_adjacent_to_child(
                        child=child, parent=parent, reverse_graph=reverse_graph
                    )
                    if depth_now < (depth_limit - 1):
                        queue.append(VertexSearchState(child, grandchildren, depth_now))
                elif edge_label not in classified_edges:
                    classified_edges.add(edge_label)
                    classification = _classify_edge(parent, child, vertex_depth, search_trees)
                    yield parent, child, classification, Direction.ALREADY_DISCOVERED, INFINITY

            if predecessor[parent]:
                yield (
                    cast(V_co, predecessor[parent]),
                    parent,
                    Label.TREE_EDGE,
                    Direction.POSTORDER,
                    vertex_depth[parent],
                )
            else:
                yield parent, parent, Label.TREE_ROOT, Direction.POSTORDER, vertex_depth[parent]


def bfs_vertex_traversal(
    graph: GraphBase[V_co, E_co],
    source: Optional[VertexType] = None,
    depth_limit: Optional[float] = None,
    reverse_graph: bool = False,
) -> Iterator[V_co]:
    """Iterates over :term:`vertices <vertex>` in a breadth-first search.

    Note:
        If ``source`` is specified, then the traversal only includes the graph
        :term:`component <connected component>` containing the ``source`` vertex.

    Note:
        Breadth-first search produces vertices in the same sequence for both :term:`preorder` and
        :term:`postorder`.

    For :term:`directed graphs <digraph>`, setting ``reverse_graph`` to True will generate
    vertices as if the graph were :term:`reversed <reverse>`.

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to begin the search. When ``source`` is
            specified, only the component reachable from the source is searched. Defaults to None.
        depth_limit: Optional; The depth limit of the search. Defaults to None (no limit).
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the :term:`reverse graph <reverse>`). Defaults to
            False.

    Yields:
        Vertex: Vertices in the breadth-first search in preorder (order of first discovery).

    Note:
        The references for this algorithm are documented in :func:`bfs_labeled_edge_traversal`.
    """
    edges = bfs_labeled_edge_traversal(
        graph, source=source, depth_limit=depth_limit, reverse_graph=reverse_graph
    )
    return (
        child for parent, child, label, direction, depth in edges if direction == Direction.PREORDER
    )


def _classify_edge(
    parent: V, child: V, vertex_depth: Dict[V, float], search_trees: UnionFind[V]
) -> str:
    """Helper function to classify non-tree edges in a breadth-first-search (BFS) tree.

    Args:
        parent: The parent vertex of the edge being classified in the BFS tree.
        child:  The child vertex of the edge being classified in the BFS tree.
        vertex_depth: A dictionary mapping vertices to their depth in the BFS tree.
        search_trees: A UnionFind data structure of disjoint sets, where each set is a BFS tree.

    Returns:
        str: An edge classification label. See
        :class:`Label <vertizee.algorithms.algo_utils.search_utils.Label>`.
    """
    if parent == child:  # self loops are considered back edges
        return Label.BACK_EDGE
    if vertex_depth[parent] == vertex_depth[child]:
        return Label.CROSS_EDGE
    if search_trees.in_same_set(parent, child):
        if vertex_depth[parent] > vertex_depth[child]:
            # parent and child are in SAME search tree and parent is an ancestor of
            # child, so its a back edge
            return Label.BACK_EDGE
        if vertex_depth[parent] < vertex_depth[child]:
            return Label.FORWARD_EDGE

        raise exception.AlgorithmError(
            "parent and child have equal depths in the "
            "breadth-first-search tree and should have been labeled as a cross edge; this error "
            "should never happen"
        )

    # parent and child are in different search trees, so its a cross edge
    return Label.CROSS_EDGE


# TODO(cpeisert): Implement and test speed versus labeled bfs traversal.
# def _plain_breadth_first_search(
#     vertices: Iterable[V], adjacency_function: Callable[[V, Optional[V]], Iterator[V]]
# ) -> Iterator[Component[V]]:
#     """Performs a plain depth-first search over the specified ``vertices``.

#     Args:
#         vertices: The graph vertices to be searched.
#         adjacency_function: The function for retrieving the adjacent vertices of each vertex during
#             the depth-first search.

#     Yields:
#         Component: An iterator of :class:`Component` objects.
#     """
#
# NOTE(cpeisert): This code is copied from _plain_depth_first_search in components.py.
#
#     seen = set()

#     for vertex in vertices:
#         if vertex in seen:
#             continue

#         component = Component(initial_vertex=vertex)
#         children = adjacency_function(vertex, None)
#         stack = [search_utils.VertexSearchState(vertex, children)]

#         while stack:
#             parent = stack[-1].parent
#             children = stack[-1].children
#             try:
#                 child = next(children)
#             except StopIteration:
#                 stack.pop()
#                 continue

#             if child not in seen:
#                 seen.add(child)
#                 component._vertices[child.label] = child
#                 grandchildren = adjacency_function(child, parent)
#                 stack.append(search_utils.VertexSearchState(child, grandchildren))

#         yield component
