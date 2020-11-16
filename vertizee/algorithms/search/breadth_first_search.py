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
from typing import Iterator, List, Optional, Set, Tuple, TYPE_CHECKING

from vertizee import VertexNotFound
from vertizee.algorithms.algo_utils.search_utils import SearchTree
from vertizee.classes.graph import GraphBase
from vertizee.classes.vertex import Vertex

if TYPE_CHECKING:
    from vertizee.classes.edge import Edge
    from vertizee.classes.vertex import VertexType


def bfs_edge_traversal(
    graph: "GraphBase",
    source: "VertexType",
    depth_limit: Optional[int] = None,
    reverse_graph: bool = False,
) -> Iterator["Edge"]:
    """Iterates over the edges of a breadth-first search traversal.

    Running time: :math:`O(|V| + |E|)`

    Note:
        The traversal only includes the graph component containing the ``source`` vertex.

    This function is adapted from the NetworkX function:
    `networkx.algorithms.traversal.breadth_first_search.generic_bfs_edges
    <https://github.com/networkx/networkx/blob/master/networkx/algorithms/traversal/breadth_first_search.py>`_
    [N2020_3]_

    The NetworkX function was in turn adapted from David Eppstein's breadth-first levels function in
    "PADS". [E2004_3]_

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
        Edge: Edges in the breadth-first-search traversal starting from ``source``.

    Example:
        The labels reveal the complete transcript of the breadth-first search algorithm.

        >>> from pprint import pprint
        >>> import vertizee as vz
        >>> from vertizee.algorithms import bfs_edge_traversal
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
        * :func:`depth_first_search`
        * :class:`SearchTree`
        * :func:`dfs_postorder_traversal`
        * :func:`dfs_preorder_traversal`
        * :class:`DepthFirstSearchResults`

    References:
     .. [E2004_3] David Eppstein's breadth-first levels function.
                  http://www.ics.uci.edu/~eppstein/PADS/BFS.py

     .. [N2020_3] NetworkX module: networkx.algorithms.traversal.breadth_first_search.py
                  https://github.com/networkx/networkx/blob/master/networkx/algorithms/traversal/breadth_first_search.py
    """
    try:
        s: Vertex = graph[source]
    except KeyError:
        raise VertexNotFound("source vertex was not found in the graph")

    if depth_limit is None:
        depth_limit = graph.vertex_count

    seen: Set[Vertex] = {s}
    queue = deque({s})

    while len(queue) > 0:



    for v in vertices:
        if v.attr[COLOR] != WHITE:  # already discovered
            continue
        adj_vertices = v.get_adj_for_search(parent=v.attr[PREDECESSOR], reverse_graph=reverse_graph)
        stack: List[_StackFrame] = [_StackFrame(v, adj_vertices, depth_limit)]

        while stack:
            v = stack[-1].vertex
            adj_vertices = stack[-1].adj_vertices
            depth_now = stack[-1].stack_frame_depth

            if v.attr[COLOR] == WHITE:  # Discovered new vertex v.
                v.attr[COLOR] = GRAY  # Mark discovered.
                timer.increment()
                v.attr[TIME_DISCOVERED] = timer.time
                if not v.attr[PREDECESSOR]:
                    yield v, v, "tree_root", "preorder"
                else:
                    yield v.attr[PREDECESSOR], v, "tree_edge", "preorder"

            if adj_vertices:  # Continue breadth-first search with next adjacent vertex.
                w = adj_vertices.pop()

                if w.attr[COLOR] == WHITE:  # Undiscovered vertex w adjacent to v.
                    w.attr[PREDECESSOR] = v
                    w_adj_vertices = w.get_adj_for_search(parent=v, reverse_graph=reverse_graph)
                    if depth_now > 1:
                        stack.append(_StackFrame(w, w_adj_vertices, depth_now - 1))
                elif w.attr[COLOR] == GRAY:  # w already discovered (still visiting adj. vertices)
                    yield v, w, "back_edge", "already_discovered"
                elif w.attr[COLOR] == BLACK:  # w already discovered (adj. visitation finished)
                    if v.attr[TIME_DISCOVERED] < w.attr[TIME_DISCOVERED]:
                        yield v, w, "forward_edge", "already_discovered"
                    else:
                        yield v, w, "cross_edge", "already_discovered"
            elif v.attr[COLOR] != BLACK:  # v is finished (adj. vertices visited - mark complete)
                stack.pop()
                v.attr[COLOR] = BLACK
                timer.increment()
                v.attr[TIME_FINISHED] = timer.time
                if not v.attr[PREDECESSOR]:
                    yield v, v, "tree_root", "postorder"
                else:
                    yield v.attr[PREDECESSOR], v, "tree_edge", "postorder"


def breadth_first_search(
    graph: "GraphBase", source: "VertexType", reverse_graph: bool = False
) -> SearchTree:
    """Performs a breadth-first-search and returns the resulting breadth-first search tree.

    Running time: :math:`O(|V| + |E|)`

    Note:
        The traversal only includes the graph component containing the ``source`` vertex.

    Note:
        This is adapted from BFS [CLRS2009_12]_.

    Args:
        graph: The graph to search.
        source: The source vertex from which to discover reachable vertices.
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.

    Returns:
        SearchTree: The breadth-first search tree formed by the search.

    Example:
        >>> import vertizee as vz
        >>> from vertizee.algorithms import depth_first_search
        >>> g = vz.Graph()
        >>> g.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 5), (6, 7)])
        >>> dfs_results = depth_first_search(g)
        >>> dfs_results.vertices_preorder
        [3, 1, 0, 2, 5, 4, 7, 6]
        >>> [str(edge) for edge in dfs_results.edges_in_discovery_order]
        ['(1, 3)', '(0, 1)', '(1, 2)', '(3, 5)', '(4, 5)', '(6, 7)']
        >>> dfs_results.is_acyclic()
        False

    See Also:
        * :class:`SearchTree
          <vertizee.algorithms.algo_utils.search_utils.SearchTree>`
        * :func:`bfs_edge_traversal`

    References:
     .. [CLRS2009_12] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                      Introduction to Algorithms: Third Edition, page 595. The MIT Press, 2009.
    """
    try:
        s: Vertex = graph[source]
    except KeyError:
        raise VertexNotFound("source vertex was not found in the graph")
    vertex_to_path_map: VertexDict[ShortestPath] = VertexDict()

    for v in graph:
        vertex_path = ShortestPath(s, v, initial_length=INFINITY, save_paths=save_paths)
        vertex_to_path_map[v] = vertex_path
    vertex_to_path_map[s].reinitialize(initial_length=0)

    # pylint: disable=unused-argument
    def weight_function(v1: Vertex, v2: Vertex, reverse_graph: bool) -> float:
        return 1

    seen: Set[Vertex] = {s}
    queue = deque({s})
    while len(queue) > 0:
        u = queue.popleft()
        u_adj = u.get_adj_for_search()
        for w in u_adj:
            if w not in seen:
                seen.add(w)
                u_path: ShortestPath = vertex_to_path_map[u]
                w_path: ShortestPath = vertex_to_path_map[w]
                w_path.relax_edge(u_path, weight_function=weight_function)
                queue.append(w)
    return vertex_to_path_map
