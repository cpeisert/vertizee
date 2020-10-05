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

"""Algorithms for depth-first graph search."""

from __future__ import annotations
from typing import Iterator, List, Optional, Set, Tuple, TYPE_CHECKING

from vertizee.algorithms.algo_utilities.depth_first_search_utils import (
    DepthFirstSearchResults,
    DepthFirstSearchTree
)
from vertizee.classes.edge import EdgeType
from vertizee.classes.graph_base import GraphBase
from vertizee.classes.vertex import Vertex

if TYPE_CHECKING:
    from vertizee.classes.vertex import VertexType

# Internal constants
BLACK = "black"
GRAY = "gray"
WHITE = "white"

COLOR = "__dfs_color"
PARENT = "__dfs_parent"
TIME_DISCOVERED = "__dfs_time_discovered"
TIME_FINISHED = "__dfs_time_finished"

NEG_INF = float("-inf")


class _StackFrame:
    def __init__(self, vertex: Vertex, adj_vertices: Set[Vertex], current_depth: int = None):
        self.vertex = vertex
        self.adj_vertices = adj_vertices
        self.stack_frame_depth = current_depth


class _Timer:
    """Class to track the relative times when vertices were first discovered and last visited."""

    def __init__(self):
        self.time: int = 0

    def increment(self):
        self.time += 1


def depth_first_search(
    graph: "GraphBase", source: Optional["VertexType"] = None, reverse_graph: bool = False
) -> DepthFirstSearchResults:
    """Performs a depth-first-search and provides detailed results (e.g. a forest of
    depth-first-search trees, cycle detection, and edge classification).

    Running time: :math:`O(|V| + |E|)`

    If a ``source`` is not specified then vertices are repeatedly selected until all components in
    the graph have been searched.

    Note:
        This is adapted from DFS and DFS-VISIT [CLRS2009_9]_, except that this implementation does
        not use recursion, thus avoiding potential stack overflow for large graphs.

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to discover reachable vertices.
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
          <vertizee.algorithms.algo_utilities.depth_first_search_utils.DepthFirstSearchResults>`
        * :class:`DepthFirstSearchTree
          <vertizee.algorithms.algo_utilities.depth_first_search_utils.DepthFirstSearchTree>`
        * :func:`dfs_postorder_traversal`
        * :func:`dfs_preorder_traversal`
        * :func:`dfs_labeled_edge_traversal`

    References:
     .. [CLRS2009_9] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                   Introduction to Algorithms: Third Edition, page 604. The MIT Press, 2009.
    """
    _initialize_dfs_graph(graph)
    dfs_results = DepthFirstSearchResults(graph)
    timer = _Timer()

    if source is None:
        vertices = graph.vertices
    else:
        vertices = {graph[source]}

    for v in vertices:
        if v.attr[COLOR] == WHITE:
            current_dfs_tree = DepthFirstSearchTree(root=v)
            dfs_results.dfs_forest.add(current_dfs_tree)
            _dfs_on_tree(
                graph=graph,
                dfs_tree=current_dfs_tree,
                dfs_results=dfs_results,
                timer=timer,
                reverse_graph=reverse_graph,
            )

    return dfs_results


def dfs_labeled_edge_traversal(
    graph: "GraphBase",
    source: Optional["VertexType"] = None,
    depth_limit: Optional[int] = None,
    reverse_graph: bool = False,
) -> Iterator[Tuple["Vertex", "Vertex", str, str]]:
    """Iterates over the labeled edges of a depth-first search traversal.

    Running time: :math:`O(|V| + |E|)`

    This function is adapted from the NetworkX function:
    `networkx.algorithms.traversal.depth_first_search.dfs_labeled_edges
    <https://github.com/networkx/networkx/blob/master/networkx/algorithms/traversal/depth_first_search.py>`_
    [N2020]_

    The NetworkX function was in turn adapted from David Eppstein's depth-first search function in
    `PADS`. [E2004]_

    For directed graphs, setting ``reverse_graph`` to True will generate edges as if the graph
    were reversed (i.e. all directed edges pointing in the opposite direction).

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to discover reachable
            vertices. Defaults to None.
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

            1. dfs_tree_root - (u, u), where u is the root vertex of a DFS tree.
            2. tree_edge - edge (u, v) is a tree edge if v was first discovered by exploring edge
               (u, v).
            3. back_edge - back edge (u, v) connects vertex u to ancestor v in a depth-first tree.
            4. forward_edge: non-tree edges (u, v) connecting a vertex u to a descendant v in a
               depth-first tree.
            5. cross_edge - All other edges, which may go between vertices in the same depth-first
               tree as long as one vertex is not an ancestor of the other, or they go between
               vertices in different depth-first trees.

        In an undirected graph, every edge is either a tree edge or a back edge.

        The ``search_direction`` is the direction of traversal and is one of the strings:

            1. preorder - the traversal discovered new vertex `child` in the DFS.
            2. postorder - the traversal finished visiting vertex `child` in the DFS.
            3. already_discovered - the traversal found a non-tree edge that had already been
               discovered.

    Example:
        The labels reveal the complete transcript of the depth-first search algorithm.

        >>> from pprint import pprint
        >>> import vertizee as vz
        >>> from vertizee.algorithms import dfs_labeled_edge_traversal
        >>> g = vz.DiGraph([(0, 1), (1, 2), (2, 1)])
        >>> pprint(list(dfs_labeled_edge_traversal(g, source=0)))
        [(0, 0, 'dfs_tree_root', 'preorder'),
         (0, 1, 'tree_edge', 'preorder'),
         (1, 2, 'tree_edge', 'preorder'),
         (2, 1, 'back_edge', 'already_discovered'),
         (1, 2, 'tree_edge', 'postorder'),
         (0, 1, 'tree_edge', 'postorder'),
         (0, 0, 'dfs_tree_root', 'postorder')]

    See Also:
        * :func:`depth_first_search`
        * :class:`DepthFirstSearchTree`
        * :func:`dfs_postorder_traversal`
        * :func:`dfs_preorder_traversal`
        * :class:`DepthFirstSearchResults`

    References:
     .. [E2004] David Eppstein's depth-first search function.
                http://www.ics.uci.edu/~eppstein/PADS

     .. [N2020] NetworkX Python package: networkx.algorithms.traversal.depth_first_search.py
                https://github.com/networkx/networkx/blob/master/networkx/algorithms/traversal/depth_first_search.py
    """
    _initialize_dfs_graph(graph)
    timer = _Timer()

    if source is None:
        vertices = graph.vertices
    else:
        vertices = {graph[source]}
    if depth_limit is None:
        depth_limit = len(graph.vertices)

    for v in vertices:
        if v.attr[COLOR] != WHITE:  # already discovered
            continue
        adj_vertices = v.get_adj_for_search(parent=v.attr[PARENT], reverse_graph=reverse_graph)
        stack: List[_StackFrame] = [_StackFrame(v, adj_vertices, depth_limit)]

        while stack:
            v = stack[-1].vertex
            adj_vertices = stack[-1].adj_vertices
            depth_now = stack[-1].stack_frame_depth

            if v.attr[COLOR] == WHITE:  # Discovered new vertex v.
                v.attr[COLOR] = GRAY  # Mark discovered.
                timer.increment()
                v.attr[TIME_DISCOVERED] = timer.time
                if not v.attr[PARENT]:
                    yield v, v, "dfs_tree_root", "preorder"
                else:
                    yield v.attr[PARENT], v, "tree_edge", "preorder"

            if adj_vertices:  # Continue depth-first search with next adjacent vertex.
                w = adj_vertices.pop()

                if w.attr[COLOR] == WHITE:  # Undiscovered vertex w adjacent to v.
                    w.attr[PARENT] = v
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
                if not v.attr[PARENT]:
                    yield v, v, "dfs_tree_root", "postorder"
                else:
                    yield v.attr[PARENT], v, "tree_edge", "postorder"


def dfs_postorder_traversal(
    graph: "GraphBase",
    source: Optional["VertexType"] = None,
    depth_limit: Optional[int] = None,
    reverse_graph: bool = False,
) -> Iterator["Vertex"]:
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
        source: Optional; The source vertex from which to discover reachable
            vertices. Defaults to None.
        depth_limit: Optional; The depth limit of the search. Defaults to None (no limit).
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.

    Yields:
        Vertex: Vertices in the depth-first search in postorder.

    See Also:
        * :func:`depth_first_search`
        * :class:`DepthFirstSearchTree`
        * :func:`dfs_preorder_traversal`
        * :func:`dfs_labeled_edge_traversal`
        * :class:`DepthFirstSearchResults`
    """
    edges = dfs_labeled_edge_traversal(
        graph, source=source, depth_limit=depth_limit, reverse_graph=reverse_graph
    )
    return (child for parent, child, label, direction in edges if direction == "postorder")


def dfs_preorder_traversal(
    graph: "GraphBase",
    source: Optional["VertexType"] = None,
    depth_limit: Optional[int] = None,
    reverse_graph: bool = False,
) -> Iterator["Vertex"]:
    """Iterates over vertices in depth-first search preorder (time of first discovery).

    For directed graphs, setting ``reverse_graph`` to True will generate vertices as if the graph
    were reversed (i.e. all directed edges pointing in the opposite direction).

    The reverse of a directed graph is also called the transpose or the converse. See
    https://en.wikipedia.org/wiki/Transpose_graph.

    Args:
        graph: The graph to search.
        source: Optional; The source vertex from which to discover reachable
            vertices. Defaults to None.
        depth_limit: Optional; The depth limit of the search. Defaults to None (no limit).
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.

    Yields:
        Vertex: Vertices in the depth-first search in preorder.

    See Also:
        * :func:`depth_first_search`
        * :class:`DepthFirstSearchTree`
        * :func:`dfs_postorder_traversal`
        * :func:`dfs_labeled_edge_traversal`
        * :class:`DepthFirstSearchResults`
    """
    edges = dfs_labeled_edge_traversal(
        graph, source=source, depth_limit=depth_limit, reverse_graph=reverse_graph
    )
    return (child for parent, child, label, direction in edges if direction == "preorder")


def _check_for_parallel_edge_cycle(graph: GraphBase, dfs_results: DepthFirstSearchResults, edge: EdgeType):
    """DFS helper function to check for parallel edge cycles."""
    if edge is None:
        return
    if not graph.is_directed_graph() and edge.parallel_edge_count > 0:
        dfs_results._is_acyclic = False
    elif graph.is_directed_graph() and dfs_results.is_acyclic():
        # Check if parallel edge in opposite direction.
        if graph[edge.vertex2, edge.vertex1] is not None:
            dfs_results._is_acyclic = False


def _dfs_on_tree(
    graph: GraphBase,
    dfs_tree: DepthFirstSearchTree,
    dfs_results: DepthFirstSearchResults,
    timer: _Timer,
    reverse_graph: bool = False,
):
    """Helper function to perform depth-first search for one DFS tree rooted at vertex
    `dfs_tree.root`.

    Args:
        graph (GraphBase): The graph to search.
        dfs_tree (DepthFirstSearchTree): A new tree to populate by performing a depth-first search
            starting at vertex `dfs_tree.root`.
        dfs_results (DepthFirstSearchResults): The object to store the results.
        timer (_Timer): _Timer to track when vertices are first discovered (or first visited) and
            the time when the visits are finished (i.e. the time after all adjacent vertices have
            been recursively visited).
        reverse_graph (bool, optional): For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.
    """
    v: Vertex = dfs_tree.root
    adj_vertices = v.get_adj_for_search(parent=v.attr[PARENT], reverse_graph=reverse_graph)
    stack: List[_StackFrame] = [_StackFrame(v, adj_vertices)]

    while stack:
        v = stack[-1].vertex
        adj_vertices = stack[-1].adj_vertices

        if v.attr[COLOR] == WHITE:  # DISCOVERED new vertex v.
            _mark_vertex_discovered_and_update(
                new_vertex=v, dfs_tree=dfs_tree, dfs_results=dfs_results, timer=timer
            )

            edge: EdgeType = _get_tree_edge_to_parent(graph, v)
            if edge is not None:
                dfs_results.edges_in_discovery_order.append(edge)
                dfs_tree.edges_in_discovery_order.append(edge)
                _check_for_parallel_edge_cycle(graph=graph, dfs_results=dfs_results, edge=edge)

        if adj_vertices:  # CONTINUE depth-first search with next adjacent vertex.
            _push_next_undiscovered_adj_vertex_to_stack(
                graph=graph,
                stack=stack,
                prev_vertex=v,
                dfs_results=dfs_results,
                reverse_graph=reverse_graph,
            )
        elif v.attr[COLOR] != BLACK:  # FINISHED visiting vertex v.
            stack.pop()
            v.attr[COLOR] = BLACK
            timer.increment()
            v.attr[TIME_FINISHED] = timer.time
            dfs_results.vertices_post_order.append(v)


def _get_tree_edge_to_parent(graph: GraphBase, v: Vertex) -> Optional[EdgeType]:
    """DFS helper function to retrieve the DFS tree edge leading from `v` to its parent vertex."""
    if v.attr[PARENT]:
        parent_v = v.attr[PARENT]
        return graph[parent_v, v]
    return None


def _initialize_dfs_graph(graph):
    """Initialize vertex attributes associated with depth-first search."""
    for v in graph.vertices:
        v.attr[COLOR] = WHITE
        v.attr[PARENT] = None
        v.attr[TIME_DISCOVERED] = NEG_INF
        v.attr[TIME_FINISHED] = NEG_INF


def _mark_vertex_discovered_and_update(
    new_vertex: Vertex, dfs_tree: DepthFirstSearchTree, dfs_results: DepthFirstSearchResults, timer: _Timer
):
    """Helper function to process newly discovered vertex."""
    new_vertex.attr[COLOR] = GRAY  # Mark discovered.
    timer.increment()
    new_vertex.attr[TIME_DISCOVERED] = timer.time

    dfs_results.vertices_pre_order.append(new_vertex)
    dfs_tree.vertices.add(new_vertex)
    if len(new_vertex.loops) > 0:
        dfs_results._is_acyclic = False


def _push_next_undiscovered_adj_vertex_to_stack(
    graph: GraphBase,
    stack: List[_StackFrame],
    prev_vertex: Vertex,
    dfs_results: DepthFirstSearchResults,
    reverse_graph: bool = False,
):
    """Helper function to update the stack with the next undiscovered adjacent vertex."""
    adj_vertices = stack[-1].adj_vertices
    w = adj_vertices.pop()
    edge = graph[prev_vertex, w]

    if w.attr[COLOR] == WHITE:  # UNDISCOVERED vertex w adjacent to v.
        dfs_results.tree_edges.add(edge)
        w.attr[PARENT] = prev_vertex
        w_adj_vertices = w.get_adj_for_search(parent=w.attr[PARENT], reverse_graph=reverse_graph)
        stack.append(_StackFrame(w, w_adj_vertices))
    elif w.attr[COLOR] == GRAY:  # ALREADY DISCOVERED, but still in the process of being visited.
        dfs_results.back_edges.add(edge)
        dfs_results._is_acyclic = False
    elif w.attr[COLOR] == BLACK:  # ALREADY DISCOVERED, and finished being visited.
        if prev_vertex.attr[TIME_DISCOVERED] < w.attr[TIME_DISCOVERED]:
            dfs_results.forward_edges.add(edge)
        else:
            dfs_results.cross_edges.add(edge)
