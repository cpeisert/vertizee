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

"""Utility functions and classes for working with shortest paths.

* :func:`reconstruct_path` - Reconstructs the shortest path between two vertices based on the
  predecessors stored in the shortest paths dictionary (or a dictionary of shortest paths
  dictionaries).
* :class:`ShortestPath` - A data structure representing a shortest path between a source vertex
  and a destination vertex.

See Also:
    * :func:`all_pairs_shortest_paths_floyd_warshall
      <vertizee.algorithms.shortest_paths.weighted.all_pairs_shortest_paths_floyd_warshall>`
    * :func:`all_pairs_shortest_paths_johnson
      <vertizee.algorithms.shortest_paths.weighted.all_pairs_shortest_paths_johnson>`
    * :func:`all_pairs_shortest_paths_johnson_fibonacci
      <vertizee.algorithms.shortest_paths.weighted.all_pairs_shortest_paths_johnson_fibonacci>`
    * :func:`shortest_paths_breadth_first_search
      <vertizee.algorithms.shortest_paths.unweighted.shortest_paths_breadth_first_search>`
    * :func:`shortest_paths_bellman_ford
      <vertizee.algorithms.shortest_paths.weighted.shortest_paths_bellman_ford>`
    * :func:`shortest_paths_dijkstra
      <vertizee.algorithms.shortest_paths.weighted.shortest_paths_dijkstra>`
    * :func:`shortest_paths_dijkstra_fibonacci
      <vertizee.algorithms.shortest_paths.weighted.shortest_paths_dijkstra_fibonacci>`
"""

from __future__ import annotations
from typing import Callable, cast, List, Optional, TYPE_CHECKING, Union

# pylint: disable=cyclic-import
from vertizee.exception import AlgorithmError
from vertizee.classes.data_structures.vertex_dict import VertexDict

if TYPE_CHECKING:
    from vertizee.classes.graph_base import GraphBase
    from vertizee.classes.vertex import Vertex, VertexType

INFINITY = float("inf")


def reconstruct_path(
    source: VertexType,
    destination: VertexType,
    paths: Union[VertexDict["ShortestPath"], VertexDict[VertexDict["ShortestPath"]]],
) -> List[Vertex]:
    """Reconstructs the shortest path between two vertices based on the predecessors stored in the
    shortest paths dictionary (or a dictionary of shortest paths dictionaries).

    Results from algorithms solving the single-source-shortest-paths problem produce a dictionary
    mapping vertices to :class:`ShortestPath` objects. Results from algorithms solving the
    all-pairs-shortest-paths problem produce a dictionary mapping source vertices to dictionaries
    mapping destination vertices to :class:`ShortestPath` objects.

    Args:
        source: The starting vertex in the path to be reconstructed.
        destination: The last vertex in the path to be reconstructed.
        paths: A dictionary of shortest paths (e.g. results from the single-source-shortest-paths
            problem) or a dictionary of dictionaries of shortest paths (e.g. results from the
            all-pairs-shortest-paths problem).

    Raises:
        AlgorithmError: An algorithm error is raised if one of the shortest path objects does not
            have a source vertex matching ``source``.

    Returns:
        List[Vertex]: The list of vertices comprising the path ``source`` :math:`\\leadsto`
        ``destination``, or an empty list if no such path exists.
    """
    if source in paths:
        if isinstance(paths[source], VertexDict):
            path_dict = cast(VertexDict["ShortestPath"], paths[source])
    if not path_dict:
        path_dict = cast(VertexDict["ShortestPath"], paths)

    path = []
    v: Optional[VertexType] = destination
    while v is not None:
        path_v: Optional[ShortestPath] = path_dict.get(v, None)
        if path_v is None:
            break
        if path_v.source != source:
            raise AlgorithmError(
                "the ShortestPath object in the paths dictionary for "
                f'vertex "{v}" does not have source vertex "{source}"'
            )
        path.append(path_v.destination)
        v = path_v.predecessor

    if len(path) > 1:
        path = list(reversed(path))
    if len(path) > 0 and path[0] != source:  # Check if a paths exists from source to destination.
        path = []
    return path


class ShortestPath:
    """Data structure representing a shortest path between a source vertex and a destination
    vertex.

    At a minimum, a shortest path includes:

        1) source - the starting vertex, designated as :math:`s`
        2) destination - the last vertex, designated as :math:`d`
        3) predecessor - the vertex immediately preceding the destination vertex, designated as
           :math:`d.predecessor`
        4) length - the path length from source to destination, designated :math:`d.length`

    In addition, a shortest path may optionally include the list of vertices comprising the path.
    The default is to save memory space by not storing the path. When paths are not saved, they
    can be calculated as needed by calling :func:`reconstruct_path`.

    The convention in graph theory is to set the initial path length, :math:`d.length`, to
    infinity (:math:`\\infty`), indicating that the destination is not reachable from the source
    (that is, unless a path is found). During the execution of a shortest-path algorithm,
    :math:`d.length` serves as an upper bound on a potential shortest path.

    Let :math:`\\delta(s,\\ d)` be the shortest-path distance from the source :math:`s` to
    destination :math:`d`.

    **Upper-bound property**:

        :math:`\\forall d \\in G(V),\\ d.length \\geq \\delta(s,\\ d)`, and once
        :math:`d.length` achieves the value :math:`\\delta(s,\\ d)` it never changes.

    **Operations that update a shortest path**:

        1) edge relaxation
        2) subpath relaxation

    Edge Relaxation
        The process of edge relaxation is testing whether one of the destination's incoming edges
        provides a shorter path from the source to the destination than the current estimate.

        Formally, let :math:`s` be the source vertex and :math:`(c,\\ d)` be an incoming edge
        to the destination vertex :math:`d`. Define :math:`w` to be a weight function, such that
        :math:`w(u,\\ v)` returns the weight (i.e. length) of some :math:`(u,\\ v)` edge.

        Then edge relaxation is defined as:

        | RELAX :math:`(c,\\ d,\\ w)`
        | |emsp| :math:`if\\ d.length \\gt c.length + w(c,\\ d)`
        | |emsp| |emsp| :math:`d.length = c.length + w(c,\\ d)`
        | |emsp| |emsp| :math:`d.predecessor = c`

    Subpath Relaxation
        Subpath relaxation is the process of testing whether some intermediate vertex :math:`k`
        yields two subpaths, that when combined, provide a shorter path from the source to
        destination than the current estimate :math:`d.length`.

        Formally, let :math:`s \\leadsto k` be a path connecting the source :math:`s` to some vertex
        :math:`k`, and let :math:`k \\leadsto d` be a path from :math:`k` to the destination
        :math:`d`. Define the path length function :math:`\\ell(x \\leadsto y)` (script 'L') to be
        the function returning the length of some path from :math:`x` to :math:`y`.

        | RELAX-SUBPATHS :math:`(s \\leadsto k,\\ k \\leadsto d)`
        | |emsp| :math:`if\\ d.length \\gt \\ell(s,\\ k) + \\ell(k,\\ d)`
        | |emsp| |emsp| :math:`d.length = \\ell(s,\\ k) + \\ell(k,\\ d)`
        | |emsp| |emsp| :math:`d.predecessor = predecessor(k \\leadsto d)`

    Args:
        source: The source vertex of the path.
        destination: The destination vertex to which a path is to be found.
        initial_length: Optional; Sets the initial path length. Defaults to infinity.
        add_initial_s_d_edge: Optional; If True, then if the initial length is less than
            infinity and there exists an edge connecting source to destination, the edge is
            added to the path. Defaults to True.
    """

    def __init__(
        self,
        source: "Vertex",
        destination: "Vertex",
        initial_length: float = INFINITY,
        save_paths: bool = False,
        add_initial_s_d_edge: bool = True,
    ):
        if source is None:
            raise ValueError("source was NoneType")
        if destination is None:
            raise ValueError("destination was NoneType")
        self._source: Vertex = source
        self._destination: Vertex = destination

        self._edge_count: int = 0
        self._length: float = initial_length
        self._predecessor: Optional[Vertex] = None
        self._store_full_path = save_paths

        if self._store_full_path:
            self._path: List[Vertex] = [self.source]
        else:
            self._path = []

        if add_initial_s_d_edge:
            self._add_s_d_edge_if_exists()

    @property
    def destination(self) -> "Vertex":
        """The destination vertex of the path."""
        return self._destination

    @property
    def edge_count(self) -> int:
        """The number of edges comprising the path."""
        return self._edge_count

    def is_destination_reachable(self) -> bool:
        """Returns True if the destination vertex is reachable from the source vertex."""
        return self._length < INFINITY

    @property
    def length(self) -> float:
        """The length of this path from the source to the destination vertex. If there is no
        path connecting source to destination, then the length is infinity."""
        return self._length

    @property
    def path(self) -> List["Vertex"]:
        """The list of vertices comprising the path (only set if ``save_paths`` is
        initialized to True).

        If ``save_paths`` was set to False upon initialization (the default), then this list
        will always be empty. However, the path can be calculated using the function
        :func:`reconstruct_path`.

        See Also:
            :func:`reconstruct_path`
        """
        if self._length == INFINITY:
            return []
        return self._path.copy()

    @property
    def predecessor(self) -> Optional["Vertex"]:
        """The vertex immediately preceding the destination vertex."""
        return self._predecessor

    def reinitialize(
        self, initial_length: float = INFINITY, add_initial_s_d_edge: bool = True
    ) -> None:
        """Reinitializes the shortest path by setting the initial length and clearing the
        ``path`` vertex list.

        Args:
            initial_length: Optional; Sets the initial path length. Defaults to infinity.
            add_initial_s_d_edge: Optional; If True, then if the initial length is less than
                infinity and there exists an edge connecting source to destination, the edge is
                added to the path. Defaults to True.
        """
        self._length = initial_length
        self._edge_count = 0

        if self._store_full_path:
            self._path = [self.source]
        else:
            self._path = []

        if add_initial_s_d_edge:
            self._add_s_d_edge_if_exists()

    def relax_edge(
        self,
        predecessor_path: "ShortestPath",
        weight_function: Callable[["Vertex", "Vertex", bool], float],
        reverse_graph: bool = False,
    ) -> bool:
        """Tests whether there is a shorter path from the source vertex through ``predecessor_path``
        to this destination vertex via an edge (predecessor_path.destination, self.destination),
        and if so, updates this path to go through the predecessor path.

        This method returns False unless the following conditions are met:

         - This path and the predecessor path share the same source vertex.
         - There exists an edge (predecessor_path.destination, self.destination), or if
           ``reverse_graph`` is True, there exists an edge
           (self.destination, predecessor_path.destination)

        See Also:
            :class:`ShortestPath`

        Args:
            predecessor_path: A graph path from the source to some vertex, such
                that there exists and edge (predecessor_path.destination, self.destination).
            weight_function: A function that accepts two vertices and a boolean indicating if the
                graph is reversed (i.e. edges of directed graphs in the opposite direction) and
                returns the corresponding edge weight. If not provided, then the default
                ``Edge.weight`` property is used. For multigraphs, the lowest edge weight among the
                parallel edges is used.
            reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
                as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults
                to False.

        Returns:
            bool: Returns True if the path was relaxed (i.e. shortened).
        """
        # Check if this path is the zero-length source.
        if self.source == self.destination:
            return False
        if self.source != predecessor_path.source:
            return False
        # If the predecessor path has infinite length, then it will not help shorten this path.
        if predecessor_path.length == INFINITY:
            return False

        j: Vertex = predecessor_path.destination
        k: Vertex = self.destination

        edge_len = weight_function(j, k, reverse_graph)
        if edge_len is None:  # This can happen if the weight function is designed as a filter.
            return False

        if self.length > predecessor_path.length + edge_len:
            self._edge_count += 1
            self._length = predecessor_path.length + edge_len
            self._predecessor = predecessor_path.destination

            if self._store_full_path:
                self._path = predecessor_path.path
                self._path.append(self.destination)

            return True

        return False

    def relax_subpaths(self, path_s_k: "ShortestPath", path_k_d: "ShortestPath") -> bool:
        """Tests whether some vertex :math:`k` yields two subpaths, that when combined, provide a
        shorter path from the source :math:`s` to destination :math:`d` than the current estimate
        :math:`d.length`, and if so, updates the path by combing the two subpaths.

        This method returns False unless the following conditions are met:

         - The source of path :math:`s \\leadsto k` is equal to ``self.source``.
         - The destination of path :math:`s \\leadsto k` is equal to the source of path
           :math:`k \\leadsto d`

        See Also:
            :class:`ShortestPath`

        Args:
            path_s_k: A graph path from the source to some vertex :math:`k`.
            path_k_d: A graph path from some vertex :math:`k` to the destination vertex.

        Returns:
            bool: Returns True if the path was relaxed (i.e. shortened).
        """
        if self.length <= (path_s_k.length + path_k_d.length):
            return False
        if path_s_k.source != self.source:
            return False
        if path_s_k.destination != path_k_d.source:
            return False

        # Merge paths s ~> k and k ~> d.
        self._edge_count = path_s_k.edge_count + path_k_d.edge_count
        self._length = path_s_k.length + path_k_d.length
        self._predecessor = path_k_d.predecessor

        if self._store_full_path:
            self._path = path_s_k.path
            if len(path_k_d._path) > 1:
                self._path += path_k_d.path[1:]

        return True

    @property
    def source(self) -> "Vertex":
        """The source vertex of the path."""
        return self._source

    def _add_s_d_edge_if_exists(self) -> None:
        """If the length is less then infinity and there exists an edge connecting source to
        destination, then the edge is added to the path and predecessor is set to source."""
        if self._length < INFINITY and self._source != self._destination:
            graph = self._source._parent_graph
            edge = graph.get_edge(self._source, self._destination)
            if edge is not None:
                self._edge_count += 1
                self._predecessor = self._source
                if self._store_full_path:
                    self._path.append(self._destination)
