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
* :class:`DepthFirstSearchTree` - A tree comprised of vertices and edges discovered during a
  depth-first search.
"""

from __future__ import annotations
from typing import List, Optional, Set

from vertizee.classes.edge import EdgeType
from vertizee.classes.graph_base import GraphBase
from vertizee.classes.vertex import Vertex


class DepthFirstSearchResults:
    """Stores the results of a depth-first search.

    A depth-first search produces the following output:

        * A forest of depth-first-search trees.
        * An ordering of vertices sorted from last to first finishing time. The finishing time of a
          vertex is the time at which the search of the vertex's subtree finished.
        * Topological sort: If the graph is a DAG (directed, acyclic), then reverse postordering
          of the vertices is a topological sort.
        * Cycle detection: For both directed and undirected graphs, if there is a back edge, then
          the graph has a cycle and the state ``is_acyclic`` is set to False. A cycle
          is a path (with at least one edge) whose first and last vertices are the same. The
          minimal cycle is a self loop and second smallest cycle is two vertices connected by
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
        dfs_forest: A depth-first forest, which is a set of :class:`DepthFirstSearchTree` objects.
        edges_in_discovery_order: The edges in the order traversed by the depth-first search.
        tree_edges: The set of tree edges.
        vertices_pre_order: The list of vertices in ascending order of first discovery times during
            the DFS.
        vertices_post_order: The list of vertices in descending order of discovery finishing time.
            The finishing time is the time at which all of the paths incident to the vertex have
            been fully explored. For directed, acyclic graphs, the reverse postorder is a
            topological sort.

    See Also:
        * :func:`depth_first_search
          <vertizee.algorithms.search.depth_first_search.depth_first_search>`
        * :class:`DepthFirstSearchTree`
        * :func:`dfs_preorder_traversal
          <vertizee.algorithms.search.depth_first_search.dfs_preorder_traversal>`
        * :func:`dfs_postorder_traversal
          <vertizee.algorithms.search.depth_first_search.dfs_postorder_traversal>`
        * :func:`dfs_labeled_edge_traversal
          <vertizee.algorithms.search.depth_first_search.dfs_labeled_edge_traversal>`
    """

    def __init__(self, graph: "GraphBase") -> None:
        self.graph: GraphBase = graph
        self.edges_in_discovery_order: List[EdgeType] = []
        self.dfs_forest: Set["DepthFirstSearchTree"] = set()

        # Edge classification.
        self.back_edges: Set[EdgeType] = set()
        self.cross_edges: Set[EdgeType] = set()
        self.forward_edges: Set[EdgeType] = set()
        self.tree_edges: Set[EdgeType] = set()

        self._is_acyclic = True

        self.vertices_pre_order: List[Vertex] = []
        self.vertices_post_order: List[Vertex] = []

    def get_topological_sort(self) -> Optional[List["Vertex"]]:
        """Returns a list of topologically sorted vertices, or None if the graph is not a directed
        acyclic graph (DAG).

        Note:
            The topological ordering is the reverse of the depth-first search postordering. The
            reverse of the postordering is not the same as the preordering.
        """
        if self.graph.is_directed_graph() and self.is_acyclic():
            return list(reversed(self.vertices_post_order))
        return None

    def is_acyclic(self) -> bool:
        """Returns True if the graph cycle free (i.e. does not contain cycles)."""
        return self._is_acyclic


class DepthFirstSearchTree:
    """A depth-first tree is a tree comprised of vertices and edges discovered during a
    depth-first search.

    Attributes:
        root: Optional; The root vertex of the DFS tree.
        edges_in_discovery_order: The edges in the order traversed by the depth-first search of the
            tree.
        vertices: The set of vertices visited during the depth-first search.
    """

    def __init__(self, root: Optional["Vertex"] = None) -> None:
        self.root = root
        self.edges_in_discovery_order: List[EdgeType] = []
        self.vertices: Set[Vertex] = set()
        if root is not None:
            self.vertices.add(root)
