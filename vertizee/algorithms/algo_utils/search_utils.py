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

"""Utility classes supporting graph search.

* :class:`SearchResults` - Stores the results of a graph search.
* :class:`SearchTree` - A tree comprised of vertices and edges discovered during a
  graph search.
"""

from __future__ import annotations
from typing import Final, Generic, List, Iterator, Optional, Set

from vertizee import exception
from vertizee.classes import primitives_parsing
from vertizee.classes.collection_views import ListView, SetView
from vertizee.classes.graph import E, GraphBase, V
from vertizee.classes.primitives_parsing import GraphPrimitive, ParsedEdgeAndVertexData


class SearchResults(Generic[V, E]):
    """Stores the results of a graph search.

    A graph search produces the following output:

        * A forest of search trees.
        * Preorder: An ordering of the vertices sorted by first to last time of discovery. The time
          of discovery is when a vertex is first found during a search.
        * Postorder: An ordering of vertices sorted by first to last finishing time. The
          finishing time is the time at which the search of the vertex's subtree (DFS) or adjacent
          neighbors (BFS) is finished.
        * Topological ordering: If the graph is a DAG (directed, acyclic), then the a depth-first
          search produces and a postordering, that when reversed, is a topological ordering.
        * Cycle detection: When a depth-first search is performend, then cycles are detected in
          both directed and undirected graphs. A cycle is a path (with at least one edge) whose
          first and last vertices are the same. The minimal cycle is a self loop and the second
          smallest cycle is two vertices connected by parallel edges (in the case of a directed
          graph, the parallel edges must be facing opposite directions).
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
        * :class:`SearchTree`
    """

    def __init__(self, graph: GraphBase[V, E], depth_first_search: bool) -> None:
        self._depth_first_search = depth_first_search
        self._graph = graph
        self._edges_in_discovery_order: List[E] = []
        self._search_tree_forest: Set[SearchTree[V, E]] = set()

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

    def graph_search_trees(self) -> SetView[SearchTree[V, E]]:
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the the
        graph search trees found during the graph search."""
        return SetView(self._search_tree_forest)

    def has_topological_ordering(self) -> bool:
        """Returns True if the search results provide a valid topological ordering of the vertices.

        A topological ordering is only ensured if the following three conditions are met:

        - The graph is directed.
        - The graph is acyclic (no cycles).
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
        in a topological ordering.

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


class SearchTree(Generic[V, E]):
    """A search tree is a tree comprised of vertices and edges discovered during a breadth-first
    or depth-first search.

    Args:
        root: The root vertex of the search tree.
    """

    def __init__(self, root: V) -> None:
        self._edges_in_discovery_order: List[E] = list()
        self._vertices_in_discovery_order: List[V] = list()

        self._edge_set: Set[E] = set()
        self._vertex_set: Set[V] = set()

        self._vertex_set.add(root)
        self._vertices_in_discovery_order.append(root)

    def __contains__(self, edge_or_vertex: GraphPrimitive) -> bool:
        if not self._vertices_in_discovery_order:
            return False

        vertex = self._vertices_in_discovery_order[0]
        graph = vertex._parent_graph
        data: ParsedEdgeAndVertexData = primitives_parsing.parse_graph_primitive(edge_or_vertex)

        if data.edges:
            if graph.has_edge(data.edges[0].vertex1.label, data.edges[0].vertex2.label):
                edge = graph[data.edges[0].vertex1.label, data.edges[0].vertex2.label]
                return edge in self._edge_set
            return False
        if data.vertices:
            return data.vertices[0].label in self._vertex_set

        raise exception.VertizeeException("expected GraphPrimitive (EdgeType or VertexType); found "
            f"{type(edge_or_vertex).__name__}")

    def __iter__(self) -> Iterator[V]:
        """Iterates over the vertices of the search tree in discovery order."""
        yield from self._vertices_in_discovery_order

    def __len__(self) -> int:
        """Returns the number of vertices in the search tree when the built-in Python function
        ``len`` is used."""
        return len(self._vertex_set)

    def edges_in_discovery_order(self) -> ListView[E]:
        """Returns a :class:`ListView <vertizee.classes.collection_views.ListView>` of the edges in
        the search tree in order of discovery."""
        return ListView(self._edges_in_discovery_order)

    @property
    def root(self) -> V:
        """The root vertex of the search tree."""
        return self._vertices_in_discovery_order[0]

    def vertices_in_discovery_order(self) -> ListView[V]:
        """Returns a :class:`ListView <vertizee.classes.collection_views.ListView>` of the vertices
        in the search tree in order of discovery."""
        return ListView(self._vertices_in_discovery_order)

    def _add_edge(self, edge: E) -> None:
        """Adds a new edge to the search tree. Exactly one of the edge's vertices should already
        be in the search tree, since otherwise, the edge would be unreachable from the existing
        tree.

        Args:
            edge: The edge to add.

        Raises:
            AlgorithmError: If exactly one of the edge's vertices is not already in the search tree.
        """
        if edge in self._edge_set:
            return
        if edge.vertex1 not in self._vertex_set and edge.vertex2 not in self._vertex_set:
            raise exception.AlgorithmError(f"neither of the edge vertices {edge} were found in the "
                "search tree; exactly one of the vertices must already be in the tree")

        self._edge_set.add(edge)
        self._edges_in_discovery_order.append(edge)

        if edge.vertex1 not in self._vertex_set:
            self._vertex_set.add(edge.vertex1)
            self._vertices_in_discovery_order.append(edge.vertex1)
        else:
            self._vertex_set.add(edge.vertex2)
            self._vertices_in_discovery_order.append(edge.vertex2)


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
