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

"""Utility classes supporting :term:`graph` search.

* :class:`Direction` - Container class for constants used to indicate the direction of traversal at
  each step of a graph search.
* :class:`Label` - Container class for constants used to label the search tree root vertices and
  edges found during a graph search.
* :class:`SearchResults` - Stores the results of a graph search.
* :class:`VertexSearchState` - A class to save the state of which adjacent vertices (``children``)
  of a vertex (``parent``) still have not been visited.
"""

from __future__ import annotations
from typing import Final, Generic, List, Iterator, Optional, Set

from vertizee import exception
from vertizee.classes.collection_views import ListView, SetView
from vertizee.classes.data_structures.tree import Tree
from vertizee.classes.edge import E
from vertizee.classes.graph import G
from vertizee.classes.vertex import V


class Direction:
    """Container class for constants used to indicate the direction of traversal at each step of a
    graph search."""

    ALREADY_DISCOVERED: Final = "already_discovered"
    """The search traversal found a non-tree edge that has already been discovered."""

    PREORDER: Final = "preorder"
    """The search traversal discovered a new vertex."""

    POSTORDER: Final = "postorder"
    """The search traversal finished visiting a vertex."""


class Label:
    """Container class for constants used to label the search tree root vertices and edges found
    during a graph search."""

    BACK_EDGE: Final = "back_edge"
    """Label for a back edge math:`(u, v)` that connects vertex math:`u` to ancestor math:`v` in a
    search tree."""

    CROSS_EDGE: Final = "cross_edge"
    """Label for a cross edge math:`(u, v)`, which may connect vertices in the same search tree (as
    long as one vertex is not an ancestor of the other), or connect vertices in different search
    trees (within a forest of search trees)."""

    FORWARD_EDGE: Final = "forward_edge"
    """Label for a forward edge math:`(u, v)` connecting a vertex math:`u` to a descendant math:`v`
    in a search tree."""

    TREE_EDGE: Final = "tree_edge"
    """Label for a tree edge math:`(u, v)`, where math:`v` was first discovered by exploring edge
    math:`(u, v)`."""

    TREE_ROOT: Final = "tree_root"
    """Label for vertex pair math:`(u, u)`, where math:`u` is the root vertex of a search tree."""


class SearchResults(Generic[V, E]):
    """Stores the results of a :term:`graph` search.

    A graph search produces the following outputs:

        * A :term:`forest` of search :term:`trees <tree>`.
        * Preorder: An ordering of the vertices sorted by first to last time of discovery. The time
          of discovery is when a vertex is first found during a search.
        * Postorder: An ordering of vertices sorted by first to last finishing time. The
          finishing time is the time at which the search of the vertex's subtree (DFS) or adjacent
          neighbors (BFS) is finished.
        * :term:`Topological ordering`: If the graph is a :term:`dag`, then a depth-first search
          produces a postordering, that when reversed, is a topological ordering.
        * :term:`Cycle` detection: When a depth-first search is performend, cycles are detected in
          both directed and undirected graphs.
        * Edge classification: The edges of a graph are classified into the following categories:

            1. Tree edges - edge :math:`(u, v)` is a tree edge if :math:`v` was first discovered by
               exploring edge :math:`(u, v)`.
            2. Back edges - back edge :math:`(u, v)` connects vertex :math:`u` to ancestor :math:`v`
               in a search tree. Per *Introduction to Algorithms*, self loops are considered back
               edges.
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
            search. False indicates that the search results are based on a bread-first search.

    See Also:
        * :func:`bfs_labeled_edge_traversal
          <vertizee.algorithms.search.breadth_first_search.bfs_labeled_edge_traversal>`
        * :func:`breadth_first_search
          <vertizee.algorithms.search.breadth_first_search.breadth_first_search>`
        * :func:`depth_first_search
          <vertizee.algorithms.search.depth_first_search.depth_first_search>`
        * :func:`dfs_preorder_traversal
          <vertizee.algorithms.search.depth_first_search.dfs_preorder_traversal>`
        * :func:`dfs_postorder_traversal
          <vertizee.algorithms.search.depth_first_search.dfs_postorder_traversal>`
        * :func:`dfs_labeled_edge_traversal
          <vertizee.algorithms.search.depth_first_search.dfs_labeled_edge_traversal>`
        * :class:`Tree <vertizee.classes.data_structures.tree.Tree>`
    """

    def __init__(self, graph: G[V, E], depth_first_search: bool) -> None:
        self._depth_first_search = depth_first_search
        self._graph = graph
        self._edges_in_discovery_order: List[E] = []
        self._search_tree_forest: Set[Tree[V, E]] = set()

        # Edge classification.
        self._back_edges: Set[E] = set()
        self._cross_edges: Set[E] = set()
        self._forward_edges: Set[E] = set()
        self._tree_edges: Set[E] = set()
        self._vertices_postorder: List[V] = []
        self._vertices_preorder: List[V] = []

        self._is_acyclic = True

    def __iter__(self) -> Iterator[V]:
        """Returns an iterator over the preorder vertices found during the graph search. The
        preorder is the order in which the vertices were discovered."""
        yield from self._vertices_preorder

    def back_edges(self) -> SetView[E]:
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the back edges
        found during the graph search."""
        return SetView(self._back_edges)

    def cross_edges(self) -> SetView[E]:
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the cross edges
        found during the graph search."""
        return SetView(self._cross_edges)

    def edges_in_discovery_order(self) -> ListView[E]:
        """Returns a :class:`ListView <vertizee.classes.collection_views.ListView>` of the edges
        found during the graph search in order of discovery."""
        return ListView(self._edges_in_discovery_order)

    def forward_edges(self) -> SetView[E]:
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the forward
        edges found during a graph search. """
        return SetView(self._forward_edges)

    def is_acyclic(self) -> bool:
        """Returns True if the graph is cycle free (i.e. does not contain cycles), otherwise False.

        Note:
            Cycle detection is only supported when using depth-first search. When using bread-first
            search, accessing this method with raise `VertizeeException`.

        Note:
            If a source vertex is specified for a depth-first search, then ``is_acyclic`` will only
            indicate if the component reachable from ``source`` contains cycles. If the component
            is acyclic, it is still possible that the graph contains another component with a cycle.

        Raises:
            VertizeeException: Raises exception if the search was not depth-first.
        """
        if not self._depth_first_search:
            raise exception.VertizeeException("cycle detection not supported using breadth-first "
                "search; use depth-first search instead")
        return self._is_acyclic

    def graph_search_trees(self) -> SetView[Tree[V, E]]:
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the the
        graph search trees found during the graph search."""
        return SetView(self._search_tree_forest)

    def has_topological_ordering(self) -> bool:
        """Returns True if the search results provide a valid :term:`topological ordering` of the
        vertices.

        A topological ordering is only ensured if the following three conditions are met:

        - The graph is directed.
        - The graph is :term:`acyclic`.
        - A depth-first search was used (as opposed to a breadth first search).
        """
        return self._depth_first_search and self._is_acyclic and self._graph.is_directed()

    def is_from_depth_first_search(self) -> bool:
        """Returns True is the search results are from a depth-first search. Returns False if the
        results are from a breadth-first search."""
        return self._depth_first_search

    def tree_edges(self) -> SetView[E]:
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the tree
        edges found during the graph search."""
        return SetView(self._tree_edges)

    def vertices_postorder(self) -> ListView[V]:
        """Returns a :class:`ListView <vertizee.classes.collection_views.ListView>` of the vertices
        in the search tree in postorder."""
        return ListView(self._vertices_postorder)

    def vertices_preorder(self) -> ListView[V]:
        """Returns a :class:`ListView <vertizee.classes.collection_views.ListView>` of the vertices
        in the search tree in preorder (i.e., in order of discovery)."""
        return ListView(self._vertices_preorder)

    def vertices_topological_order(self) -> ListView[V]:
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
            raise exception.Unfeasible("a topological ordering is only valid for a depth-first "
                f"search on a directed, acyclic graph; error: {error_msg}")
        return ListView(list(reversed(self._vertices_postorder)))


class VertexSearchState:
    """A class to save the state of which adjacent vertices (``children``) of a vertex (``parent``)
    still have not been visited.

    When searching a graph (for example, using a depth-first or bread-first search), the search
    creates a tree structure, where a parent is at a higher level in the tree than its adjacent
    children.

    Args:
        parent: The parent vertex relative to ``children`` in a search tree (e.g. a depth-first
            tree or bread-first tree).
        children: An iterator over the unvisited children of ``parent`` in a search tree. The
            child vertices are adjacent to ``parent``.
        depth: The depth of ``parent`` relative to the root of the search tree.
    """
    def __init__(
        self, parent: V, children: Iterator[V], depth: Optional[int] = None
    ) -> None:
        self.parent = parent
        self.children = children
        self.depth = depth
