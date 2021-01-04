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
==========================
Search: depth-first search
==========================

Algorithms for depth-first search. The asymptotic running times use the
notation that for some graph :math:`G(V, E)`, the number of vertices is :math:`n = |V|` and the
number of edges is :math:`m = |E|`.

**Recommended Tutorial**: :doc:`Search <../../tutorials/search>` - |image-colab-search|

.. |image-colab-search| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/cpeisert/vertizee/blob/master/docs/source/tutorials/search.ipynb

Function summary
================

* :func:`dfs` - Performs a depth-first-search and provides detailed results (e.g. a :term:`forest`
  of depth-first-search :term:`trees <rooted tree>`, :term:`cycle` detection, and edge
  classification). Running time: :math:`O(m + n)`
* :func:`dfs_labeled_edge_traversal` - Iterates over the labeled edges of a depth-first search
  traversal. Running time: :math:`O(m + n)`
* :func:`dfs_postorder_traversal` - Iterates over vertices of a depth-first search in
  :term:`postorder`. Running time: :math:`O(m + n)`
* :func:`dfs_preorder_traversal` - Iterates over vertices of depth-first search in :term:`preorder`.
  Running time: :math:`O(m + n)`

Detailed documentation
======================
"""

from __future__ import annotations
from typing import Final, Iterator, List, Optional, Set, Tuple, TYPE_CHECKING

from vertizee import exception
from vertizee.algorithms.algo_utils.search_utils import (
    Direction,
    Label,
    SearchResults,
    VertexSearchState,
)
from vertizee.classes import edge as edge_module
from vertizee.classes.data_structures.tree import Tree
from vertizee.classes.data_structures.vertex_dict import VertexDict

if TYPE_CHECKING:
    from vertizee.classes.edge import E
    from vertizee.classes.graph import G
    from vertizee.classes.vertex import V, VertexType

BLACK: Final = "black"
GRAY: Final = "gray"
WHITE: Final = "white"

INFINITY: Final = float("inf")


def dfs(
    graph: G[V, E], source: Optional[VertexType] = None, reverse_graph: bool = False
) -> SearchResults[V, E]:
    """Performs a depth-first-search and provides detailed results (e.g. a forest of
    depth-first-search trees, cycle detection, and edge classification).

    Running time: :math:`O(m + n)`

    Note:
        If a ``source`` is not specified, then vertices are repeatedly selected until all
        :term:`components <connected component>` in the graph have been searched.

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to begin the search. When ``source`` is
            specified, only the component reachable from the source is searched. Defaults to None.
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the :term:`reverse graph <reverse>`). Defaults to
            False.

    Returns:
        SearchResults: The results of the depth-first search.

    Example:
        >>> import vertizee as vz
        >>> from vertizee.algorithms import search
        >>> g = vz.Graph()
        >>> g.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 5), (6, 7)])
        >>> dfs_results = search.depth_first_search(g)
        >>> dfs_results.vertices_preorder
        [3, 1, 0, 2, 5, 4, 7, 6]
        >>> [str(edge) for edge in dfs_results.edges_in_discovery_order]
        ['(1, 3)', '(0, 1)', '(1, 2)', '(3, 5)', '(4, 5)', '(6, 7)']
        >>> dfs_results.is_acyclic()
        False

    See Also:
        * :class:`SearchResults <vertizee.algorithms.algo_utils.search_utils.SearchResults>`
        * :class:`Tree <vertizee.classes.data_structures.tree.Tree>`

    Note:
        The references for this algorithm are documented in :func:`dfs_labeled_edge_traversal`.
    """
    dfs_results = SearchResults(graph, depth_first_search=True)

    labeled_edge_tuple_iterator = dfs_labeled_edge_traversal(
        graph, source=source, reverse_graph=reverse_graph
    )

    for (parent, child, label, direction) in labeled_edge_tuple_iterator:
        vertex = child

        if direction == Direction.PREORDER:
            if vertex.loop_edge:
                dfs_results._is_acyclic = False

            if label == Label.TREE_ROOT:
                dfs_tree = Tree(root=vertex)
                dfs_results._search_tree_forest.add(dfs_tree)
                dfs_results._vertices_preorder.append(vertex)
        elif direction == Direction.POSTORDER:
            dfs_results._vertices_postorder.append(vertex)

        if label == Label.TREE_EDGE and direction == Direction.PREORDER:
            edge = graph[parent, child]
            dfs_results._tree_edges.add(edge)
            dfs_tree.add_edge(edge)
            dfs_results._edges_in_discovery_order.append(edge)
            dfs_results._vertices_preorder.append(vertex)
            if dfs_results._is_acyclic:
                _check_for_parallel_edge_cycle(graph, dfs_results, edge)
        elif label == Label.BACK_EDGE:
            dfs_results._is_acyclic = False
            dfs_results._back_edges.add(graph[parent, child])
        elif label == Label.CROSS_EDGE:
            dfs_results._cross_edges.add(graph[parent, child])
        elif label == Label.FORWARD_EDGE:
            dfs_results._forward_edges.add(graph[parent, child])

    return dfs_results


def dfs_labeled_edge_traversal(
    graph: G[V, E],
    source: Optional[VertexType] = None,
    depth_limit: Optional[int] = None,
    reverse_graph: bool = False,
) -> Iterator[Tuple[V, V, str, str]]:
    """Iterates over the labeled edges of a depth-first search traversal.

    Running time: :math:`O(m + n)`

    Note:
        If ``source`` is specified, then the traversal only includes the graph
        :term:`component <connected component>` containing the ``source`` vertex.

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to begin the search. When ``source`` is
            specified, only the component reachable from the source is searched. Defaults to None.
        depth_limit: Optional; The depth limit of the search. Defaults to None (no limit).
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the :term:`reverse graph <reverse>`). Defaults
            to False.

    Yields:
        Tuple[Vertex, Vertex, str, str]: An iterator over tuples of the form
        ``(parent, child, label, search_direction)`` where ``(parent, child)`` is the edge being
        explored in the depth-first search. The ``child`` vertex is found by iterating over the
        parent's adjacency list.

        The ``label`` is one of the strings:

            1. "tree_root" - :math:`(u, u)`, where :math:`u` is the root vertex of a DFS tree.
            2. "tree_edge" - edge :math:`(u, v)` is a tree edge if :math:`v` was first discovered by
               exploring edge :math:`(u, v)`.
            3. "back_edge" - back edge :math:`(u, v)` connects vertex :math:`u` to ancestor
               :math:`v` in a depth-first tree. Per *Introduction to Algorithms* :cite:`2009:clrs`,
               :term:`self loops <loop>` are considered back edges.
            4. "forward_edge" - non-tree edges :math:`(u, v)` connecting a vertex :math:`u` to a
               descendant :math:`v` in a depth-first tree.
            5. "cross_edge" - All other edges, which may go between vertices in the same depth-first
               tree as long as one vertex is not an ancestor of the other, or they go between
               vertices in different depth-first trees.

        In an undirected graph, every edge is either a tree edge or a back edge.

        The ``search_direction`` is the direction of traversal and is one of the strings:

            1. "preorder" - the traversal discovered new vertex `child` in the DFS.
            2. "postorder" - the traversal finished visiting vertex `child` in the DFS.
            3. "already_discovered" - the traversal found a non-tree edge connecting to a vertex
               that was already discovered.

    Example:
        The labels reveal the complete transcript of the depth-first search algorithm.

        >>> from pprint import pprint
        >>> import vertizee as vz
        >>> from vertizee.algorithms import dfs_labeled_edge_traversal
        >>> g = vz.DiGraph([(0, 1), (1, 2), (2, 1)])
        >>> pprint(list(dfs_labeled_edge_traversal(g, source=0)))
        [(0, 0, 'tree_root', 'preorder'),
         (0, 1, 'tree_edge', 'preorder'),
         (1, 2, 'tree_edge', 'preorder'),
         (2, 1, 'back_edge', 'already_discovered'),
         (1, 2, 'tree_edge', 'postorder'),
         (0, 1, 'tree_edge', 'postorder'),
         (0, 0, 'tree_root', 'postorder')]

    See Also:
        * :class:`Direction <vertizee.algorithms.algo_utils.search_utils.Direction>`
        * :class:`Label <vertizee.algorithms.algo_utils.search_utils.Label>`
        * :class:`SearchResults <vertizee.algorithms.algo_utils.search_utils.SearchResults>`

    Note:
        This function is adapted from the NetworkX function:
        `networkx.algorithms.traversal.depth_first_search.dfs_labeled_edges
        <https://github.com/networkx/networkx/blob/master/networkx/algorithms/traversal/depth_first_search.py>`_
        :cite:`2008:hss`

        The NetworkX function was in turn adapted from David Eppstein's depth-first search function
        in `PADS`. :cite:`2015:eppstein`

        The edge labeling of this function is based on the treatment in *Introduction to
        Algorithms*. :cite:`2009:clrs`

        The feature to allow depth limits is based on Korf. :cite:`1985:korf`
    """
    if len(graph) == 0:
        raise exception.Unfeasible("search is undefined for an empty graph")

    classified_edges: Set[str] = set()
    """The set of edges that have been classified so far by the depth-first search into one of:
    tree edge, back edge, cross edge, or forward edge."""

    vertex_color: VertexDict[str] = VertexDict()
    """A mapping from vertices to their color (white, gray, black) indicating the status of each
    vertex in the search process (i.e. undiscovered, in the process of being visited, or visit
    finished)."""

    vertex_discovery_order: VertexDict[int] = VertexDict()
    """A mapping from vertices to the order in which they were discovered by the depth-first
    search."""

    for vertex in graph.vertices():
        vertex_color[vertex] = WHITE

    if source is None:
        vertices = graph.vertices()
    else:
        s: V = graph[source]
        vertices = {s}
    if depth_limit is None:
        depth_limit = INFINITY

    for vertex in vertices:
        if vertex_color[vertex] != WHITE:  # Already discovered?
            continue

        vertex_color[vertex] = GRAY  # Mark discovered.
        vertex_discovery_order[vertex] = len(vertex_discovery_order)

        children = _get_adjacent_to_child(child=vertex, parent=None, reverse_graph=reverse_graph)
        stack: List[VertexSearchState] = [VertexSearchState(vertex, children, depth_limit)]

        yield vertex, vertex, Label.TREE_ROOT, Direction.PREORDER

        # Explore the depth-first search tree rooted at `vertex`.
        while stack:
            parent = stack[-1].parent
            children = stack[-1].children
            depth_now = stack[-1].depth

            try:
                child = next(children)
            except StopIteration:
                stack_frame = stack.pop()
                child = stack_frame.parent
                vertex_color[child] = BLACK  # Finished visiting child.
                if stack:
                    parent = stack[-1].parent
                    yield parent, child, Label.TREE_EDGE, Direction.POSTORDER
                else:
                    yield child, child, Label.TREE_ROOT, Direction.POSTORDER
                continue

            edge_label = edge_module.create_edge_label(parent, child, graph.is_directed())
            if vertex_color[child] == WHITE:  # Discovered new vertex?
                vertex_color[child] = GRAY  # Mark discovered and in the process of being visited.
                vertex_discovery_order[child] = len(vertex_discovery_order)
                classified_edges.add(edge_label)
                yield parent, child, Label.TREE_EDGE, Direction.PREORDER

                grandchildren = _get_adjacent_to_child(
                    child=child, parent=parent, reverse_graph=reverse_graph
                )
                if depth_now > 1:
                    stack.append(VertexSearchState(child, grandchildren, depth_now - 1))
            elif vertex_color[child] == GRAY:  # In the process of being visited?
                if edge_label not in classified_edges:
                    classified_edges.add(edge_label)
                    yield parent, child, Label.BACK_EDGE, Direction.ALREADY_DISCOVERED
            elif vertex_color[child] == BLACK:  # Finished being visited?
                if edge_label not in classified_edges:
                    classified_edges.add(edge_label)
                    if vertex_discovery_order[parent] < vertex_discovery_order[child]:
                        yield parent, child, Label.FORWARD_EDGE, Direction.ALREADY_DISCOVERED
                    else:
                        yield parent, child, Label.CROSS_EDGE, Direction.ALREADY_DISCOVERED
            else:
                raise exception.AlgorithmError(
                    f"vertex color '{vertex_color[child]}' of vertex '{child}' not recognized"
                )


def dfs_postorder_traversal(
    graph: G[V, E],
    source: Optional[VertexType] = None,
    depth_limit: Optional[int] = None,
    reverse_graph: bool = False,
) -> Iterator[V]:
    """Iterates over vertices of a depth-first search in :term:`postorder`.

    Note:
        If ``source`` is specified, then the traversal only includes the graph
        :term:`component <connected component>` containing the ``source`` vertex.

    :term:`Postorder <postorder>` is the order in which a depth-first search last visited the
    vertices. A vertex visit is finished when all of the vertex's adjacent vertices have been
    recursively visited. If the graph is directed and :term:`acyclic` (a.k.a. a :term:`dag`), then
    the reverse postorder forms a :term:`topological ordering` of the vertices (i.e. the first
    vertex returned from ``next()`` will be the last vertex in the :term:`topological sort
    <topological sorting>`).

    For directed graphs, setting `reverse_graph` to True will generate vertices as if the graph
    were :term:`reversed <reverse>`.

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to begin the search. When ``source`` is
            specified, only the component reachable from the source is searched. Defaults to None.
        depth_limit: Optional; The depth limit of the search. Defaults to None (no limit).
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal as if
            the graph were reversed (i.e. the :term:`reverse graph <reverse>`). Defaults to False.

    Yields:
        Vertex: Vertices in the depth-first search in postorder.

    Note:
        The references for this algorithm are documented in :func:`dfs_labeled_edge_traversal`.
    """
    edges = dfs_labeled_edge_traversal(
        graph, source=source, depth_limit=depth_limit, reverse_graph=reverse_graph
    )
    return (child for parent, child, label, direction in edges if direction == Direction.POSTORDER)


def dfs_preorder_traversal(
    graph: G[V, E],
    source: Optional[VertexType] = None,
    depth_limit: Optional[int] = None,
    reverse_graph: bool = False,
) -> Iterator[V]:
    """Iterates over vertices in depth-first search in :term:`preorder`.

    Note:
        If ``source`` is specified, then the traversal only includes the graph
        :term:`component <connected component>` containing the ``source`` vertex.

    For directed graphs, setting ``reverse_graph`` to True will generate vertices as if the graph
    were :term:`reversed <reverse>`.

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to begin the search. When ``source`` is
            specified, only the component reachable from the source is searched. Defaults to None.
        depth_limit: Optional; The depth limit of the search. Defaults to None (no limit).
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the :term:`reverse graph <reverse>`). Defaults to
            False.

    Yields:
        Vertex: Vertices in the depth-first search in preorder.

    Note:
        The references for this algorithm are documented in :func:`dfs_labeled_edge_traversal`.
    """
    edges = dfs_labeled_edge_traversal(
        graph, source=source, depth_limit=depth_limit, reverse_graph=reverse_graph
    )
    return (child for parent, child, label, direction in edges if direction == Direction.PREORDER)


def _check_for_parallel_edge_cycle(
    graph: G[V, E], dfs_results: SearchResults[V, E], edge: E
) -> None:
    """Helper function to check for parallel edge cycles."""
    if edge is None:
        return
    if not graph.is_directed() and graph.is_multigraph():
        if edge.multiplicity > 1:
            dfs_results._is_acyclic = False
    elif graph.is_directed() and dfs_results.is_acyclic():
        # Check if parallel edge in opposite direction.
        if graph.has_edge(edge.vertex2, edge.vertex1):
            dfs_results._is_acyclic = False


def _get_adjacent_to_child(child: V, parent: Optional[V], reverse_graph: bool) -> Iterator[V]:
    if child._parent_graph.is_directed():
        if reverse_graph:
            return iter(child.adj_vertices_incoming())
        return iter(child.adj_vertices_outgoing())

    # undirected graph
    adj_vertices = child.adj_vertices()
    if parent:
        adj_vertices = adj_vertices - {parent}
    return iter(adj_vertices)
