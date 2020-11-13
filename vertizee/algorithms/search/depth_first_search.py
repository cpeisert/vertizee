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

"""Algorithms for depth-first search."""

from __future__ import annotations
from typing import Final, Iterator, List, Optional, Set, Tuple, TYPE_CHECKING

from vertizee import exception
from vertizee.algorithms.algo_utils.search_utils import (
    DepthFirstSearchResults,
    SearchTree
)
from vertizee.classes import edge as edge_module
from vertizee.classes.graph import E, GraphBase, V
from vertizee.classes.data_structures.vertex_dict import VertexDict

if TYPE_CHECKING:
    from vertizee.classes.vertex import VertexType

# Internal constants
BLACK = "black"
GRAY = "gray"
WHITE = "white"

NEG_INF = float("-inf")


class _StackFrame:
    def __init__(
        self, parent: V, children: Iterator[V], current_depth: Optional[int] = None
    ) -> None:
        self.parent = parent
        self.children = children
        self.depth = current_depth


class Direction:
    """Container class for constants used to indicate the direction of traversal at each step of a
    depth-first search."""

    ALREADY_DISCOVERED: Final = "already_discovered"
    """The depth-first search traversal found a non-tree edge that has already been discovered."""

    PREORDER: Final = "preorder"
    """The depth-first search traversal discovered a new vertex."""

    POSTORDER: Final = "postorder"
    """The depth-first search traversal finished visiting a vertex."""


class Label:
    """Container class for constants used to label the search tree root vertices and edges found
    during a depth-first search."""

    BACK_EDGE: Final = "back_edge"
    """Label for a back edge math:`(u, v)` that connects vertex math:`u` to ancestor math:`v` in a
    depth-first tree."""

    CROSS_EDGE: Final = "cross_edge"
    """Label for a cross edge math:`(u, v)`, which may connect vertices in the same depth-first
    tree (as long as one vertex is not an ancestor of the other), or connect vertices in different
    depth-first trees (within a depth-first search forest)."""

    FORWARD_EDGE: Final = "forward_edge"
    """Label for a forward edge math:`(u, v)` connecting a vertex math:`u` to a descendant math:`v`
    in a depth-first tree."""

    TREE_EDGE: Final = "tree_edge"
    """Label for a tree edge math:`(u, v)`, where math:`v` was first discovered by exploring edge
    math:`(u, v)`."""

    TREE_ROOT: Final = "tree_root"
    """Label for vertex pair math:`(u, u)`, where math:`u` is the root vertex of a DFS tree."""


def depth_first_search(
    graph: GraphBase[V, E], source: Optional["VertexType"] = None, reverse_graph: bool = False
) -> DepthFirstSearchResults[V, E]:
    """Performs a depth-first-search and provides detailed results (e.g. a forest of
    depth-first-search trees, cycle detection, and edge classification).

    Running time: :math:`O(|V| + |E|)`

    If a ``source`` is not specified, then vertices are repeatedly selected until all components in
    the graph have been searched.

    Note:
        This is adapted from DFS and DFS-VISIT [CLRS2009_9]_, except that this implementation does
        not use recursion, thus avoiding potential stack overflow for large graphs.

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to begin the search. When ``source`` is
            specified, only the component reachable from the source is searched. Defaults to None.
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.

    Returns:
        DepthFirstSearchResults: The results of the depth-first search.

    Example:
        >>> import vertizee as vz
        >>> from vertizee.algorithms import depth_first_search
        >>> g = vz.Graph()
        >>> g.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 5), (6, 7)])
        >>> dfs_results = depth_first_search(g)
        >>> dfs_results.vertices_pre_order
        [3, 1, 0, 2, 5, 4, 7, 6]
        >>> [str(edge) for edge in dfs_results.edges_in_discovery_order]
        ['(1, 3)', '(0, 1)', '(1, 2)', '(3, 5)', '(4, 5)', '(6, 7)']
        >>> dfs_results.is_acyclic()
        False

    See Also:
        * :class:`DepthFirstSearchResults
          <vertizee.algorithms.algo_utils.search_utils.DepthFirstSearchResults>`
        * :class:`SearchTree
          <vertizee.algorithms.algo_utils.search_utils.SearchTree>`
        * :func:`dfs_postorder_traversal`
        * :func:`dfs_preorder_traversal`
        * :func:`dfs_labeled_edge_traversal`

    References:
     .. [CLRS2009_9] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                     Introduction to Algorithms: Third Edition, page 604. The MIT Press, 2009.
    """
    dfs_results = DepthFirstSearchResults(graph)

    labeled_edge_tuple_iterator = dfs_labeled_edge_traversal(
        graph, source=source, reverse_graph=reverse_graph
    )

    for (parent, child, label, direction) in labeled_edge_tuple_iterator:
        vertex = child

        if direction == Direction.PREORDER:
            if vertex.loop_edge:
                dfs_results._is_acyclic = False

            if label == Label.TREE_ROOT:
                dfs_tree = SearchTree(root=vertex)
                dfs_results._dfs_forest.add(dfs_tree)
                dfs_results._vertices_pre_order.append(vertex)
        elif direction == Direction.POSTORDER:
            dfs_results._vertices_post_order.append(vertex)

        if label == Label.TREE_EDGE and direction == Direction.PREORDER:
            edge = graph[parent, child]
            dfs_results._tree_edges.add(edge)
            dfs_tree._add_edge(edge)
            dfs_results._edges_in_discovery_order.append(edge)
            dfs_results._vertices_pre_order.append(vertex)
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
    graph: GraphBase[V, E],
    source: Optional[V] = None,
    depth_limit: Optional[int] = None,
    reverse_graph: bool = False
) -> Iterator[Tuple[V, V, str, str]]:
    """Iterates over the labeled edges of a depth-first search traversal.

    Running time: :math:`O(|V| + |E|)`

    This function is adapted from the NetworkX function:
    `networkx.algorithms.traversal.depth_first_search.dfs_labeled_edges
    <https://github.com/networkx/networkx/blob/master/networkx/algorithms/traversal/depth_first_search.py>`_
    [N2020]_

    The NetworkX function was in turn adapted from David Eppstein's depth-first search function in
    `PADS`. [E2004]_

    The edge labeling of this function is based on the treatment in *Introduction to Algorithms*.
    [CLRS2009_10]_

    For directed graphs, setting ``reverse_graph`` to True will generate edges as if the graph
    were reversed (i.e. all directed edges pointing in the opposite direction).

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to begin the search. When ``source`` is
            specified, only the component reachable from the source is searched. Defaults to None.
        depth_limit: Optional; The depth limit of the search. Defaults to None (no limit).
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults
            to False.

    Yields:
        Tuple[Vertex, Vertex, str, str]: An iterator over tuples of the form
        ``(parent, child, label, search_direction)`` where ``(parent, child)`` is the edge being
        explored in the depth-first search. The ``child`` vertex is found by iterating over the
        parent's adjacency list.

        The ``label`` is one of the strings:

            1. "tree_root" - math:`(u, u)`, where math:`u` is the root vertex of a DFS tree.
            2. "tree_edge" - edge math:`(u, v)` is a tree edge if math:`v` was first discovered by
               exploring edge math:`(u, v)`.
            3. "back_edge" - back edge math:`(u, v)` connects vertex math:`u` to ancestor math:`v`
               in a depth-first tree.
            4. "forward_edge": non-tree edges math:`(u, v)` connecting a vertex math:`u` to a
               descendant math:`v` in a depth-first tree.
            5. "cross_edge" - All other edges, which may go between vertices in the same depth-first
               tree as long as one vertex is not an ancestor of the other, or they go between
               vertices in different depth-first trees.

        In an undirected graph, every edge is either a tree edge or a back edge.

        The ``search_direction`` is the direction of traversal and is one of the strings:

            1. "preorder" - the traversal discovered new vertex `child` in the DFS.
            2. "postorder" - the traversal finished visiting vertex `child` in the DFS.
            3. "already_discovered" - the traversal found a non-tree edge that had already been
               discovered.

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
        * :func:`depth_first_search`
        * :func:`dfs_postorder_traversal`
        * :func:`dfs_preorder_traversal`
        * :class:`Direction`
        * :class:`Label`

    References:
     .. [CLRS2009_10] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                      Introduction to Algorithms: Third Edition, page 603-610. The MIT Press, 2009.

     .. [E2004] David Eppstein's depth-first search function.
                http://www.ics.uci.edu/~eppstein/PADS

     .. [N2020] NetworkX module: networkx.algorithms.traversal.depth_first_search.py
                https://github.com/networkx/networkx/blob/master/networkx/algorithms/traversal/depth_first_search.py
    """
    return _dfs_labeled_edge_traversal(
        graph, source=source, depth_limit=depth_limit, reverse_graph=reverse_graph)


def dfs_postorder_traversal(
    graph: GraphBase[V, E],
    source: Optional["VertexType"] = None,
    depth_limit: Optional[int] = None,
    reverse_graph: bool = False,
) -> Iterator[V]:
    """Iterates over vertices in depth-first search postorder, meaning the order in which vertices
    were last visited.

    Postorder is the order in which a depth-first search last visited the vertices. A vertex
    visit is finished when all of the vertex's adjacent vertices have been recursively visited. If
    the graph is directed and acyclic (a.k.a. a DAG), then the reverse postorder forms a
    topological sort of the vertices (i.e. the first vertex returned from next() will be the last
    vertex in the topological sort).

    For directed graphs, setting `reverse_graph` to True will generate vertices as if the graph
    were reversed (i.e. all directed edges pointing in the opposite direction).

    The reverse of a directed graph is also called the transpose or the converse. See
    https://en.wikipedia.org/wiki/Transpose_graph.

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to begin the search. When ``source`` is
            specified, only the component reachable from the source is searched. Defaults to None.
        depth_limit: Optional; The depth limit of the search. Defaults to None (no limit).
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.

    Yields:
        Vertex: Vertices in the depth-first search in postorder.

    See Also:
        * :func:`depth_first_search`
        * :func:`dfs_preorder_traversal`
        * :func:`dfs_labeled_edge_traversal`
        * :class:`Direction`
        * :class:`Label`
    """
    edges = dfs_labeled_edge_traversal(
        graph, source=source, depth_limit=depth_limit, reverse_graph=reverse_graph
    )
    return (child for parent, child, label, direction in edges if direction == Direction.POSTORDER)


def dfs_preorder_traversal(
    graph: GraphBase[V, E],
    source: Optional["VertexType"] = None,
    depth_limit: Optional[int] = None,
    reverse_graph: bool = False,
) -> Iterator[V]:
    """Iterates over vertices in depth-first search preorder (time of first discovery).

    For directed graphs, setting ``reverse_graph`` to True will generate vertices as if the graph
    were reversed (i.e. all directed edges pointing in the opposite direction).

    The reverse of a directed graph is also called the transpose or the converse. See
    https://en.wikipedia.org/wiki/Transpose_graph.

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to begin the search. When ``source`` is
            specified, only the component reachable from the source is searched. Defaults to None.
        depth_limit: Optional; The depth limit of the search. Defaults to None (no limit).
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.

    Yields:
        Vertex: Vertices in the depth-first search in preorder.

    See Also:
        * :func:`depth_first_search`
        * :func:`dfs_postorder_traversal`
        * :func:`dfs_labeled_edge_traversal`
    """
    edges = dfs_labeled_edge_traversal(
        graph, source=source, depth_limit=depth_limit, reverse_graph=reverse_graph
    )
    return (child for parent, child, label, direction in edges if direction == Direction.PREORDER)


def _check_for_parallel_edge_cycle(
    graph: GraphBase[V, E], dfs_results: DepthFirstSearchResults[V, E], edge: E
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


def _dfs_labeled_edge_traversal(
    graph: GraphBase[V, E],
    source: Optional[V] = None,
    depth_limit: Optional[int] = None,
    reverse_graph: bool = False,
) -> Iterator[Tuple[V, V, str, str]]:
    """Iterates over the edges of a graph in a depth-first search traversal.

    Running time: :math:`O(|V| + |E|)`

    The ``adjacency_function`` determines which adjacent vertices to return as each vertex is
    explored in the depth-first search. The default adjacency function works as follows:

        * In undirected graphs, all adjacent vertices are returned, except the parent (or
          predecessor) vertex if provided.
        * In directed graphs, if ``reverse_graph`` is False, then the outgoing adjacent vertices
          are returned. If ``reverse_graph`` is True, then the incoming adjacent vertices are
          returned.

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to begin the search. When ``source`` is
            specified, only the component reachable from the source is searched. Defaults to None.
        depth_limit: Optional; The depth limit of the search. Defaults to None (no limit).
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults
            to False.

    Yields:
        Tuple[Vertex, Vertex, str, str]: An iterator over tuples of the form
        ``(parent, child, label, search_direction)`` where ``(parent, child)`` is the edge being
        explored in the depth-first search.
    """
    if len(graph) == 0:
        raise exception.Unfeasible("search is undefined for an empty graph")

    classified_edges: Set[str] = set()
    vertex_color: VertexDict[str] = VertexDict()
    vertex_discovery_order: VertexDict[int] = VertexDict()

    for vertex in graph.vertices():
        vertex_color[vertex] = WHITE

    if source is None:
        vertices = graph.vertices()
    else:
        s: V = graph[source]
        vertices = {s}
    if depth_limit is None:
        depth_limit = graph.vertex_count

    for vertex in vertices:
        if vertex_color[vertex] != WHITE:  # Already discovered?
            continue

        parent = vertex
        vertex_color[parent] = GRAY  # Mark discovered.
        vertex_discovery_order[parent] = len(vertex_discovery_order)

        children = _get_adjacent_to_child(child=parent, parent=None, reverse_graph=reverse_graph)
        stack: List[_StackFrame] = [_StackFrame(parent, children, depth_limit)]

        yield parent, parent, Label.TREE_ROOT, Direction.PREORDER

        for parent, child, label, direction in (
            _traverse_depth_first_search_tree(
                graph=graph,
                stack=stack,
                vertex_color=vertex_color,
                vertex_discovery_order=vertex_discovery_order,
                classified_edges=classified_edges,
                reverse_graph=reverse_graph)):
            yield parent, child, label, direction


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


def _traverse_depth_first_search_tree(
    graph: GraphBase[V, E],
    stack: List[_StackFrame],
    vertex_color: VertexDict[str],
    vertex_discovery_order: VertexDict[int],
    classified_edges: Set[str],
    reverse_graph: bool,
) -> None:
    """Iterates over the edges of a depth-first search tree rooted at the parent vertex stored on
    the ``stack``.

    Running time: :math:`O(|V| + |E|)`

    Args:
        graph: The graph to search.
        stack: A list of stack frames, where each frame stores a predecessor parent vertex, its
            adjacent vertices (children in the depth-first search tree), and an optional depth
            limit.
        vertex_color: A mapping from vertices to their color (white, gray, black) indicating the
            status of each vertex in the search process (i.e. undiscovered, in the process of being
            visited, or visit finished).
        vertex_discovery_order: A mapping from vertices to the order in which they were discovered
            by the depth-first search.
        classified_edges: The set of edges that have been classified so far by the depth-first
            search into one of: tree edge, back edge, cross edge, or forward edge.
        reverse_graph: For directed graphs, setting to True will yield a traversal as if the graph
            were reversed (i.e. the reverse/transpose/converse graph).

    Yields:
        Tuple[Vertex, Vertex, str, str]: An iterator over tuples of the form
        ``(parent, child, label, search_direction)`` where ``(parent, child)`` is the edge being
        explored in the depth-first search.

    See Also:
        * :func:`dfs_labeled_edge_traversal`
        * :func:`dfs_postorder_traversal`
        * :func:`dfs_preorder_traversal`
    """
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
                child=child, parent=parent, reverse_graph=reverse_graph)
            if depth_now > 1:
                stack.append(_StackFrame(child, grandchildren, depth_now - 1))
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
                f"vertex color '{vertex_color[child]}' of vertex '{child}' not recognized")
