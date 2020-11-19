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

"""Algorithms for breadth-first search."""

from __future__ import annotations
import collections
from typing import Deque, Dict, Final, Iterator, Optional, Set, Tuple, TYPE_CHECKING

from vertizee import exception
from vertizee.algorithms.algo_utils.search_utils import (
    Direction,
    Label,
    SearchResults,
    SearchTree,
    VertexSearchState
)
from vertizee.classes.data_structures.union_find import UnionFind
from vertizee.classes import edge as edge_module

if TYPE_CHECKING:
    from vertizee.classes.graph import E, GraphBase, V
    from vertizee.classes.vertex import VertexType

INFINITY: Final = float("inf")


def bfs_labeled_edge_traversal(
    graph: GraphBase[V, E],
    source: Optional[VertexType] = None,
    depth_limit: Optional[int] = None,
    reverse_graph: bool = False
) -> Iterator[Tuple[V, V, str]]:
    """Iterates over the labeled edges of a breadth-first search traversal.

    Running time: :math:`O(|V| + |E|)`

    Note:
        If ``source`` is specified, then only vertices within the graph component containing
        ``source`` will be traversed.

    For directed graphs, setting ``reverse_graph`` to True will generate edges as if the graph
    were reversed (i.e. all directed edges pointing in the opposite direction).

    Args:
        graph: The graph to search.
        source: The source vertex from which to discover reachable vertices.
        depth_limit: Optional; The depth limit of the search. Defaults to None (no limit).
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults
            to False.

    Yields:
        Tuple[Vertex, Vertex, str, str, int]: An iterator over tuples of the form
        ``(parent, child, label, search_direction, depth)`` where ``(parent, child)`` is the edge
        being explored in the depth-first search.

        The ``label`` is one of the strings:

            1. "tree_root" - math:`(u, u)`, where math:`u` is the root vertex of a DFS tree.
            2. "tree_edge" - edge math:`(u, v)` is a tree edge if math:`v` was first discovered by
               exploring edge math:`(u, v)`.
            3. "back_edge" - back edge math:`(u, v)` connects vertex math:`u` to ancestor math:`v`
               in a depth-first tree. Per *Introduction to Algorithms* [CLRS2009_9]_, self loops
               are considered back edges.
            4. "forward_edge": non-tree edges math:`(u, v)` connecting a vertex math:`u` to a
               descendant math:`v` in a depth-first tree.
            5. "cross_edge" - All other edges, which may go between vertices in the same depth-first
               tree as long as one vertex is not an ancestor of the other, or they go between
               vertices in different depth-first trees.

        The ``search_direction`` is the direction of traversal and is one of the strings:

            1. "preorder" - the traversal discovered new vertex `child` in the DFS.
            2. "postorder" - the traversal finished visiting vertex `child` in the DFS.
            3. "already_discovered" - the traversal found a non-tree edge that had already been
               discovered.

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
        * :func:`breadth_first_search`
        * :class:`Label`

    Notes:
        This function uses ideas from the NetworkX function:
        `networkx.algorithms.traversal.breadth_first_search.generic_bfs_edges
        <https://github.com/networkx/networkx/blob/master/networkx/algorithms/traversal/breadth_first_search.py>`_
        [N2020_3]_

        The NetworkX function was in turn adapted from David Eppstein's depth-first search function
        in `PADS`. [E2004_3]_

        The edge labeling of this function is based on the treatment in *Introduction to
        Algorithms*. [CLRS2009_9]_

        The feature to allow depth limits is based on the the Wikipedia article "Iterative
        deepening depth-first search" [WEC2020_03]_.

    References:
     .. [CLRS2009_9] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                     Introduction to Algorithms: Third Edition, page 594-610. The MIT Press, 2009.

     .. [E2004_3] David Eppstein's breadth-first levels function.
                  http://www.ics.uci.edu/~eppstein/PADS/BFS.py

     .. [N2020_3] NetworkX module: networkx.algorithms.traversal.breadth_first_search.py
                  https://github.com/networkx/networkx/blob/master/networkx/algorithms/traversal/breadth_first_search.py

     .. [WEC2020_3] Wikipedia contributors. "Iterative deepening depth-first search." Wikipedia,
                    The Free Encyclopedia. Available from:
                    https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search.
                    Accessed 16 November 2020.
    """
    if len(graph) == 0:
        raise exception.Unfeasible("search is undefined for an empty graph")

    classified_edges: Set[str] = set()
    """The set of edges that have been classified so far by the breadth-first search into one of:
    tree edge, back edge, cross edge, or forward edge."""

    predecessor: Dict[V] = collections.defaultdict(lambda: None)
    """The predecessor is the parent vertex in the BFS tree. Root vertices have predecessor None.
    In addition, if a source vertex is specified, unreachable vertices also have predecessor None.
    """

    search_trees: UnionFind[V] = UnionFind()
    """UnionFind data structure, where each disjoint set contains the vertices from a breadth-first
    search tree."""

    seen: Set[V] = set()
    """A set of the vertices discovered so far during a breadth-first search."""

    vertex_depth: Dict[V, int] = collections.defaultdict(lambda: INFINITY)

    if source is None:
        vertices = graph.vertices()
    else:
        s: V = graph[source]
        vertices = {s}
    if depth_limit is None:
        depth_limit = INFINITY

    for vertex in vertices:
        if vertex in seen:
            continue

        depth_now = 0
        search_trees.make_set(vertex)  # New BFS tree root.
        seen.add(vertex)
        vertex_depth[vertex] = depth_now

        children = _get_adjacent_to_child(child=vertex, parent=None, reverse_graph=reverse_graph)
        queue: Deque[VertexSearchState] = collections.deque()
        queue.append(VertexSearchState(vertex, children, depth_now))

        yield vertex, vertex, Label.TREE_ROOT, Direction.PREORDER, depth_now

        # Explore the bread-first search tree rooted at `vertex`.
        while queue:
            parent_state: VertexSearchState = queue.popleft()
            parent = parent_state.parent

            # [START for]
            for child in parent_state.children:
                edge_label = edge_module.create_edge_label(parent, child, graph.is_directed())

                if child not in seen:  # Discovered new vertex?
                    seen.add(child)
                    depth_now = parent_state.depth + 1
                    vertex_depth[child] = depth_now
                    predecessor[child] = parent
                    search_trees.make_set(child)
                    search_trees.union(parent, child)
                    classified_edges.add(edge_label)

                    yield parent, child, Label.TREE_EDGE, Direction.PREORDER, depth_now

                    grandchildren = _get_adjacent_to_child(
                        child=child, parent=parent, reverse_graph=reverse_graph)
                    if depth_now < (depth_limit - 1):
                        queue.append(VertexSearchState(child, grandchildren, depth_now))
                elif edge_label not in classified_edges:
                    classified_edges.add(edge_label)
                    if not search_trees.in_same_set(parent, child):
                        # parent and child are in different search trees, so its a cross edge
                        yield (parent, child, Label.CROSS_EDGE, Direction.ALREADY_DISCOVERED,
                            INFINITY)
                    elif parent == child:  # self loops are considered back edges
                        yield parent, child, Label.BACK_EDGE, Direction.ALREADY_DISCOVERED, INFINITY
                    elif vertex_depth[parent] == vertex_depth[child]:
                        yield (parent, child, Label.CROSS_EDGE, Direction.ALREADY_DISCOVERED,
                            INFINITY)
                    elif vertex_depth[parent] < vertex_depth[child]:
                        yield (parent, child, Label.FORWARD_EDGE, Direction.ALREADY_DISCOVERED,
                            INFINITY)
                    else:
                        yield parent, child, Label.BACK_EDGE, Direction.ALREADY_DISCOVERED, INFINITY
            # [END for]

            if predecessor[parent]:
                yield (predecessor[parent], parent, Label.TREE_EDGE, Direction.POSTORDER,
                    vertex_depth[parent])
            else:
                yield parent, parent, Label.TREE_ROOT, Direction.POSTORDER, vertex_depth[parent]


def bfs_preorder_traversal(
    graph: GraphBase[V, E],
    source: Optional[VertexType] = None,
    depth_limit: Optional[int] = None,
    reverse_graph: bool = False,
) -> Iterator[V]:
    """Iterates over vertices in breadth-first search preorder (order of first discovery).

    For directed graphs, setting ``reverse_graph`` to True will generate vertices as if the graph
    were reversed (i.e. all directed edges pointing in the opposite direction).

    The reverse of a directed graph is also called the transpose or the converse. See
    https://en.wikipedia.org/wiki/Transpose_graph.

    Note:
        The preorder and postorder are identical for breadth-first search, which is why there is no
        function named ``bfs_postorder_traversal``.

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to begin the search. When ``source`` is
            specified, only the component reachable from the source is searched. Defaults to None.
        depth_limit: Optional; The depth limit of the search. Defaults to None (no limit).
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.

    Yields:
        Vertex: Vertices in the breadth-first search in preorder (order of first discovery).

    See Also:
        * :func:`bfs_labeled_edge_traversal`
        * :func:`breadth_first_search`

    Note:
        The references for this algorithm are documented in :func:`bfs_labeled_edge_traversal`.
    """
    edges = bfs_labeled_edge_traversal(
        graph, source=source, depth_limit=depth_limit, reverse_graph=reverse_graph
    )
    return (child for parent, child, label, direction, depth in edges
        if direction == Direction.PREORDER)


def breadth_first_search(
    graph: GraphBase[V, E], source: Optional[VertexType] = None, reverse_graph: bool = False
) -> SearchResults[V, E]:
    """Performs a breadth-first-search and provides detailed results (e.g. a forest of
    breadth-first-search trees and edge classification).

    Running time: :math:`O(|V| + |E|)`

    If a ``source`` is not specified, then vertices are repeatedly selected until all components in
    the graph have been searched.

    Note:
        Breadth-first search does not support cycle detection. For cycle detection, use
        :func:`depth_first_search
        <vertizee.algorithms.search.depth_first_search.depth_first_search>`.

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to begin the search. When ``source`` is
            specified, only the component reachable from the source is searched. Defaults to None.
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.

    Returns:
        SearchResults: The results of the depth-first search.

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
        * :class:`SearchResults
          <vertizee.algorithms.algo_utils.search_utils.SearchResults>`
        * :class:`SearchTree
          <vertizee.algorithms.algo_utils.search_utils.SearchTree>`
        * :func:`bfs_labeled_edge_traversal`

    Note:
        The references for this algorithm are documented in :func:`bfs_labeled_edge_traversal`.
    """
    results = SearchResults(graph, depth_first_search=False)

    labeled_edge_tuple_iterator = bfs_labeled_edge_traversal(
        graph, source=source, reverse_graph=reverse_graph
    )

    for (parent, child, label, direction, _) in labeled_edge_tuple_iterator:
        vertex = child

        if direction == Direction.PREORDER:
            if label == Label.TREE_ROOT:
                bfs_tree = SearchTree(root=vertex)
                results._search_tree_forest.add(bfs_tree)
                results._vertices_preorder.append(vertex)
        elif direction == Direction.POSTORDER:
            results._vertices_postorder.append(vertex)

        if label == Label.TREE_EDGE and direction == Direction.PREORDER:
            edge = graph[parent, child]
            results._tree_edges.add(edge)
            bfs_tree._add_edge(edge)
            results._edges_in_discovery_order.append(edge)
            results._vertices_preorder.append(vertex)
        elif label == Label.BACK_EDGE:
            results._back_edges.add(graph[parent, child])
        elif label == Label.CROSS_EDGE:
            results._cross_edges.add(graph[parent, child])
        elif label == Label.FORWARD_EDGE:
            results._forward_edges.add(graph[parent, child])

    return results


def _get_adjacent_to_child(
    child: V, parent: Optional[V], reverse_graph: bool
) -> Iterator[V]:
    if child._parent_graph.is_directed():
        if reverse_graph:
            return iter(child.adj_vertices_incoming())
        return iter(child.adj_vertices_outgoing())

    # undirected graph
    adj_vertices = child.adj_vertices()
    if parent:
        adj_vertices = adj_vertices - {parent}

    return iter(adj_vertices)
