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

"""Utility classes supporting depth-first graph search.

* :class:`DepthFirstSearchResults` - Stores the results of a depth-first search.
* :class:`SearchTree` - A tree comprised of vertices and edges discovered during a
  depth-first search.
"""

from __future__ import annotations
from typing import Generic, List, Iterator, Set

from vertizee import exception
from vertizee.classes import primitives_parsing
from vertizee.classes.collection_views import ListView, SetView
from vertizee.classes.graph import E, GraphBase, V
from vertizee.classes.primitives_parsing import GraphPrimitive, ParsedEdgeAndVertexData


class DepthFirstSearchResults(Generic[V, E]):
    """Stores the results of a depth-first search.

    A depth-first search produces the following output:

        * A forest of depth-first search trees.
        * An ordering of vertices sorted from last to first finishing time. The finishing time of a
          vertex is the time at which the search of the vertex's subtree finished.
        * Topological sort: If the graph is a DAG (directed, acyclic), then the reverse postordering
          of the vertices is a topological sort.
        * Cycle detection: For both directed and undirected graphs, if there is a back edge, then
          the graph has a cycle and the state ``is_acyclic`` is set to False. A cycle
          is a path (with at least one edge) whose first and last vertices are the same. The
          minimal cycle is a self loop and the second smallest cycle is two vertices connected by
          parallel edges (in the case of a directed graph, the parallel edges must be facing
          opposite directions).
        * Edge classification: The edges of the graph are classified into the following categories:

            1. Tree edges - edge :math:`(u, v)` is a tree edge if :math:`v` was first discovered by
               exploring edge :math:`(u, v)`.
            2. Back edges - back edge :math:`(u, v)` connects vertex :math:`u` to ancestor :math:`v`
               in a depth-first tree.
            3. Forward edges: non-tree edges :math:`(u, v)` connecting a vertex :math:`u` to a
               descendant :math:`v` in a depth-first tree.
            4. Cross edges - All other edges, which may go between vertices in the same depth-first
               tree as long as one vertex is not an ancestor of the other, or they go between
               vertices in different depth-first trees.

            In an undirected graph, every edge is either a tree edge or a back edge.

    Args:
        graph: The graph that was searched.

    Attributes:
        back_edges: The set of back edges.
        cross_edges: The set of cross edges.
        forward_edges: The set of forward edges.
        graph: The graph that was searched.
        dfs_forest: A depth-first forest, which is a set of :class:`SearchTree` objects.
        edges_in_discovery_order: The edges in the order traversed by the depth-first search.
        tree_edges: The set of tree edges.
        vertices_preorder: The list of vertices in ascending order of first discovery times during
            the DFS.
        vertices_postorder: The list of vertices in descending order of discovery finishing time.
            The finishing time is the time at which all of the paths incident to the vertex have
            been fully explored. For directed, acyclic graphs, the reverse postorder is a
            topological sort.

    See Also:
        * :func:`depth_first_search
          <vertizee.algorithms.search.depth_first_search.depth_first_search>`
        * :class:`SearchTree`
        * :func:`dfs_preorder_traversal
          <vertizee.algorithms.search.depth_first_search.dfs_preorder_traversal>`
        * :func:`dfs_postorder_traversal
          <vertizee.algorithms.search.depth_first_search.dfs_postorder_traversal>`
        * :func:`dfs_labeled_edge_traversal
          <vertizee.algorithms.search.depth_first_search.dfs_labeled_edge_traversal>`
    """

    def __init__(self, graph: GraphBase[V, E]) -> None:
        self._graph = graph
        self._edges_in_discovery_order: List[E] = []
        self._dfs_forest: Set[SearchTree[V, E]] = set()

        # Edge classification.
        self._back_edges: Set[E] = set()
        self._cross_edges: Set[E] = set()
        self._forward_edges: Set[E] = set()
        self._tree_edges: Set[E] = set()
        self._vertices_postorder: List[V] = []
        self._vertices_preorder: List[V] = []

        self._is_acyclic = True

    def __iter__(self) -> Iterator[V]:
        """Returns an iterator over the preorder vertices found during the depth-first search. The
        preorder is the order in which the vertices were discovered."""
        yield from self._vertices_preorder

    def back_edges(self) -> SetView[E]:
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the back edges
        found during the depth-first search."""
        return SetView(self._back_edges)

    def cross_edges(self) -> SetView[E]:
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the cross edges
        found during the depth-first search."""
        return SetView(self._cross_edges)

    def depth_first_search_trees(self) -> SetView[SearchTree[V, E]]:
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the the
        depth-first search trees found during the depth-first search."""
        return SetView(self._dfs_forest)

    def edges_in_discovery_order(self) -> ListView[E]:
        """Returns a :class:`ListView <vertizee.classes.collection_views.ListView>` of the edges
        found during the depth-first search in order of discovery."""
        return ListView(self._edges_in_discovery_order)

    def forward_edges(self) -> SetView[E]:
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the forward
        edges found during the depth-first search."""
        return SetView(self._forward_edges)

    def is_acyclic(self) -> bool:
        """Returns True if the graph cycle free (i.e. does not contain cycles)."""
        return self._is_acyclic

    def topological_sort(self) -> Iterator[V]:
        """Returns an iterator over the topologically sorted vertices. If the graph is not directed
        and acyclic, the list will be empty.

        Note:
            The topological ordering is the reverse of the depth-first search postordering. The
            reverse of the postordering is not the same as the preordering.
        """
        if self._graph.is_directed() and self.is_acyclic():
            yield from reversed(self._vertices_postorder)
        yield from ()

    def tree_edges(self) -> SetView[E]:
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the tree
        edges found during the depth-first search."""
        return SetView(self._tree_edges)

    def vertices_postorder(self) -> ListView[V]:
        """Returns a :class:`ListView <vertizee.classes.collection_views.ListView>` of the vertices
        in the depth-first tree in postorder."""
        return ListView(self._vertices_postorder)

    def vertices_preorder(self) -> ListView[V]:
        """Returns a :class:`ListView <vertizee.classes.collection_views.ListView>` of the vertices
        in the depth-first tree in preorder (i.e., in order of discovery)."""
        return ListView(self._vertices_preorder)


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
        """Iterates over the vertices of the depth-first search tree in discovery order."""
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
