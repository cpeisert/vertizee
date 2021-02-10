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

"""
==========================
Search algorithm utilities
==========================

Utility classes supporting :term:`graph` search.

Class summary
=============

* :class:`Direction` - Container class for constants used to indicate the direction of traversal at
  each step of a graph search.
* :class:`Label` - Container class for constants used to label the search :term:`tree` root
  :term:`vertices <vertex>` and :term:`edges <edge>` found during a graph search.
* :class:`SearchResults` - Stores the results of a graph search.
* :class:`VertexSearchState` - A class to save the state of a graph search indicating for some
  *parent* vertex in a search :term:`tree`, which :term:`adjacent` vertices (*children*) have not
  yet been visited.

See Also:
    * :func:`bfs <vertizee.algorithms.search.breadth_first_search.bfs>`
    * :func:`bfs_labeled_edge_traversal
      <vertizee.algorithms.search.breadth_first_search.bfs_labeled_edge_traversal>`
    * :func:`bfs_vertex_traversal
      <vertizee.algorithms.search.breadth_first_search.bfs_vertex_traversal>`
    * :func:`dfs <vertizee.algorithms.search.depth_first_search.dfs>`
    * :func:`dfs_labeled_edge_traversal
      <vertizee.algorithms.search.depth_first_search.dfs_labeled_edge_traversal>`
    * :func:`dfs_postorder_traversal
      <vertizee.algorithms.search.depth_first_search.dfs_postorder_traversal>`
    * :func:`dfs_preorder_traversal
      <vertizee.algorithms.search.depth_first_search.dfs_preorder_traversal>`
    * :class:`Tree <vertizee.classes.data_structures.tree.Tree>`

Detailed documentation
======================
"""

from __future__ import annotations
from typing import cast, Final, Generic, List, Iterator, Optional, Set, TYPE_CHECKING

from vertizee import exception
from vertizee.classes.collection_views import ListView, SetView
from vertizee.classes.data_structures.tree import Tree
from vertizee.classes.edge import EdgeBase
from vertizee.classes.vertex import DiVertex, MultiDiVertex, V, V_co

if TYPE_CHECKING:
    from vertizee.classes.graph import GraphBase


def get_adjacent_to_child(child: V, parent: Optional[V], reverse_graph: bool) -> Iterator[V]:
    """Helper method to retrieve the vertices adjacent to `child` in the context of a graph search.
    In :term:`directed graphs <digraph>`, the adjacent vertices are the :term:`head` vertices of
    the outgoing edges. In :term:`undirected graphs <undirected graph>`, the adjacenet vertices
    are all of the adjacent vertices, excluding the `parent` vertex, where the `parent` vertex is
    the vertex from which the `child` vertex was discovered.

    Args:
        child: The vertex whose adjacent vertices are to be returned.
        parent: Optional; The `parent` vertex in the search tree from which the `child` vertex was
            discovered.
        reverse_graph: If True, then in directed graphs, the adjacent incoming vertices are
            returned rather than the outgoing adjacent vertices.

    Returns:
        Iterator[V]: An iterator over the adjacent vertices.
    """
    if child._parent_graph.is_directed():
        assert isinstance(child, (DiVertex, MultiDiVertex))
        if reverse_graph:
            return cast(Iterator[V], iter(child.adj_vertices_incoming()))
        return cast(Iterator[V], iter(child.adj_vertices_outgoing()))

    # undirected graph
    adj_vertices = set(child.adj_vertices())
    if parent:
        adj_vertices = adj_vertices - {parent}
    return cast(Iterator[V], iter(adj_vertices))


class Direction:
    """Container class for constants used to indicate the direction of traversal at each step of a
    graph search."""

    ALREADY_DISCOVERED: Final[str] = "already_discovered"
    """The search traversal found a non-tree edge connecting to a vertex that was already
    discovered."""

    PREORDER: Final[str] = "preorder"
    """The search traversal discovered a new vertex."""

    POSTORDER: Final[str] = "postorder"
    """The search traversal finished visiting a vertex."""


class Label:
    """Container class for constants used to label the search tree root vertices and edges found
    during a graph search."""

    BACK_EDGE: Final[str] = "back_edge"
    """Label for a back edge :math:`(u, v)` that connects vertex :math:`u` to ancestor :math:`v`
    in a search tree."""

    CROSS_EDGE: Final[str] = "cross_edge"
    """Label for a cross edge :math:`(u, v)`, which may connect vertices in the same search tree (as
    long as one vertex is not an ancestor of the other), or connect vertices in different search
    trees (within a forest of search trees)."""

    FORWARD_EDGE: Final[str] = "forward_edge"
    """Label for a forward edge :math:`(u, v)` connecting a vertex :math:`u` to a descendant
    :math:`v` in a search tree."""

    TREE_EDGE: Final[str] = "tree_edge"
    """Label for a tree edge :math:`(u, v)`, where :math:`v` was first discovered by exploring edge
    :math:`(u, v)`."""

    TREE_ROOT: Final[str] = "tree_root"
    """Label for vertex :math:`u`, where :math:`u` is the root vertex of a search tree. Root
    vertices are often returned as a redundant pair such as :math:`(u, u)` to provide a consistent
    format relative to back, cross, forward, and tree edges."""


class SearchResults(Generic[V_co]):
    """Stores the results of a :term:`graph` search.

    A graph search produces the following outputs:

        * A :term:`forest` of search :term:`trees <tree>`.
        * Preorder - An ordering of the vertices sorted by first to last time of discovery. The time
          of discovery is when a vertex is first found during a search.
        * Postorder - An ordering of vertices sorted by first to last finishing time. The
          finishing time is the time at which the search of the vertex's subtree (in the case of a
          depth-first search) or :term:`adjacent` neighbors (in the case of a breadth-first search)
          is finished.
        * :term:`Topological ordering <topological ordering>` - If the graph is a :term:`dag`, then
          a depth-first search produces a postordering, that when reversed, is a topological
          ordering.
        * :term:`Cycle <cycle>` detection: When a depth-first search is performend, cycles are
          detected in both directed and undirected graphs.
        * Edge classification - The edges of a graph are classified into the following categories:

          1. Tree edges - edge :math:`(u, v)` is a tree edge if :math:`v` was first discovered by
             exploring edge :math:`(u, v)`.
          2. Back edges - back edge :math:`(u, v)` connects vertex :math:`u` to ancestor :math:`v`
             in a search tree. Per *Introduction to Algorithms*, self loops are considered back
             edges. :cite:`2009:clrs`
          3. Forward edges: non-tree edges :math:`(u, v)` connecting a vertex :math:`u` to a
             descendant :math:`v` in a search tree.
          4. Cross edges - All other edges, which may go between vertices in the same search
             tree as long as one vertex is not an ancestor of the other, or they go between
             vertices in different search trees.

    Note:
        If a source vertex is specified for a depth-first search, then ``is_acyclic`` will only
        indicate if the component reachable from ``source`` contains cycles. If the component
        is acyclic, it is still possible that the graph contains another component with a cycle.

    Args:
        graph: The graph that was searched.
        depth_first_search: True indicates that the search results are based on a depth-first
            search. False indicates that the search results are based on a breadth-first search.
    """

    def __init__(self, graph: "GraphBase[V_co]", depth_first_search: bool) -> None:
        self._depth_first_search = depth_first_search
        self._graph = graph
        self._edges_in_discovery_order: List[EdgeBase[V_co]] = []
        self._search_tree_forest: Set[Tree[V_co]] = set()

        # Edge classification.
        self._back_edges: Set[EdgeBase[V_co]] = set()
        self._cross_edges: Set[EdgeBase[V_co]] = set()
        self._forward_edges: Set[EdgeBase[V_co]] = set()
        self._tree_edges: Set[EdgeBase[V_co]] = set()
        self._vertices_postorder: List[V_co] = []
        self._vertices_preorder: List[V_co] = []

        self._is_acyclic = True

    def __iter__(self) -> Iterator["V_co"]:
        """Returns an iterator over the preorder vertices found during the graph search. The
        preorder is the order in which the vertices were discovered."""
        yield from self._vertices_preorder

    def back_edges(self) -> "SetView[EdgeBase[V_co]]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the back edges
        found during the graph search."""
        return SetView(self._back_edges)

    def cross_edges(self) -> "SetView[EdgeBase[V_co]]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the cross edges
        found during the graph search."""
        return SetView(self._cross_edges)

    def edges_in_discovery_order(self) -> "ListView[EdgeBase[V_co]]":
        """Returns a :class:`ListView <vertizee.classes.collection_views.ListView>` of the edges
        found during the graph search in order of discovery."""
        return ListView(self._edges_in_discovery_order)

    def forward_edges(self) -> "SetView[EdgeBase[V_co]]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the forward
        edges found during a graph search."""
        return SetView(self._forward_edges)

    def is_acyclic(self) -> bool:
        """Returns True if the graph is cycle free (i.e. does not contain cycles), otherwise False.

        Note:
            Cycle detection is only supported when using depth-first search. When using
            breadth-first search, calling this method with raise :class:`VertizeeException
            <vertizee.exception.VertizeeException>`.

        Note:
            If a source vertex is specified for a depth-first search, then ``is_acyclic`` will only
            indicate if the component reachable from ``source`` contains cycles. If the component
            is acyclic, it is still possible that the graph contains another component with a cycle.

        Raises:
            VertizeeException: Raises exception if the search was not depth-first.
        """
        if not self._depth_first_search:
            raise exception.VertizeeException(
                "cycle detection not supported using breadth-first "
                "search; use depth-first search instead"
            )
        return self._is_acyclic

    def graph_search_trees(self) -> "SetView[Tree[V_co]]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the the
        graph search trees found during the graph search."""
        return SetView(self._search_tree_forest)

    def has_topological_ordering(self) -> bool:
        """Returns True if the search results provide a valid :term:`topological ordering` of the
        vertices.

        A topological ordering is only ensured if the following three conditions are met:

        - The graph is directed.
        - The graph is :term:`acyclic`.
        - A depth-first search was used (as opposed to a breadth-first search).
        """
        return self._depth_first_search and self._is_acyclic and self._graph.is_directed()

    def is_from_depth_first_search(self) -> bool:
        """Returns True is the search results are from a depth-first search. Returns False if the
        results are from a breadth-first search."""
        return self._depth_first_search

    def tree_edges(self) -> "SetView[EdgeBase[V_co]]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the tree
        edges found during the graph search."""
        return SetView(self._tree_edges)

    def vertices_postorder(self) -> "ListView[V_co]":
        """Returns a :class:`ListView <vertizee.classes.collection_views.ListView>` of the vertices
        in the search tree in postorder."""
        return ListView(self._vertices_postorder)

    def vertices_preorder(self) -> "ListView[V_co]":
        """Returns a :class:`ListView <vertizee.classes.collection_views.ListView>` of the vertices
        in the search tree in preorder (i.e., in order of discovery)."""
        return ListView(self._vertices_preorder)

    def vertices_topological_order(self) -> "ListView[V_co]":
        """Returns a :class:`ListView <vertizee.classes.collection_views.ListView>` of the vertices
        in a :term:`topological ordering`.

        Note:
            The topological ordering is the reverse of the depth-first search postordering. The
            reverse of the postordering is not the same as the preordering.

        Raises:
            Unfeasible: Raises ``Unfeasible`` if the conditions required to determine a topological
                ordering are not met. See :meth:`has_topological_ordering`.
        """
        if not self.has_topological_ordering():
            bfs_error = "breadth-first search used" if not self._depth_first_search else ""
            dir_error = "graph not directed" if not self._graph.is_directed() else ""
            cycle_error = "graph has cycle" if not self._is_acyclic else ""
            errors = [bfs_error, dir_error, cycle_error]
            error_msg = "; ".join(e for e in errors if e)
            raise exception.Unfeasible(
                "a topological ordering is only valid for a depth-first "
                f"search on a directed, acyclic graph; error: {error_msg}"
            )
        return ListView(list(reversed(self._vertices_postorder)))


class VertexSearchState(Generic[V_co]):
    """A class to save the state of a graph search indicating for some *parent* vertex in a
    search :term:`tree`, which :term:`adjacent` vertices (*children*) have not yet been visited.

    Note:
        When searching a graph (for example, using a depth-first or breadth-first search), the
        search creates a tree structure, where a parent is at a higher level in the tree than its
        adjacent children.

    Args:
        parent: The parent vertex relative to ``children`` in a search tree (e.g. a depth-first
            tree or breadth-first tree).
        children: An iterator over the unvisited children of ``parent`` in a search tree. The
            child vertices are adjacent to ``parent``.
        depth: The depth of ``parent`` relative to the root of the search tree.
    """

    def __init__(self, parent: V_co, children: Iterator[V_co], depth: Optional[int] = None) -> None:
        self.parent = parent
        self.children = children
        self.depth = depth
