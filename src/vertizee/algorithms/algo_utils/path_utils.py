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
==============
Path utilities
==============

Utility classes and functions for working with :term:`paths <path>`.

Class summary
=============

* :class:`ShortestPath` - A data structure representing a shortest :term:`path` between a source
  vertex and a destination vertex.

Function summary
================

* :func:`reconstruct_path` - Reconstructs the shortest :term:`path` between two
  :term:`vertices <vertex>` based on the predecessors stored in the shortest paths dictionary (or
  a dictionary of shortest paths dictionaries).

See Also:
    * :func:`all_pairs_shortest_paths
      <vertizee.algorithms.paths.all_pairs.all_pairs_shortest_paths>`
    * :func:`bellman_ford <vertizee.algorithms.paths.single_source.bellman_ford>`
    * :func:`breadth_first_search_shortest_paths
      <vertizee.algorithms.paths.single_source.breadth_first_search_shortest_paths>`
    * :func:`dijkstra <vertizee.algorithms.paths.single_source.dijkstra>`
    * :func:`dijkstra_fibonacci
      <vertizee.algorithms.paths.single_source.dijkstra_fibonacci>`
    * :func:`floyd_warshall <vertizee.algorithms.paths.all_pairs.floyd_warshall>`
    * :func:`johnson <vertizee.algorithms.paths.all_pairs.johnson>`
    * :func:`johnson_fibonacci
      <vertizee.algorithms.paths.all_pairs.johnson_fibonacci>`
    * :func:`shortest_paths <vertizee.algorithms.paths.single_source.shortest_paths>`

Detailed documentation
======================
"""

from __future__ import annotations
from typing import Callable, cast, Final, Generic, List, Optional, Union

from vertizee import exception
from vertizee.classes.collection_views import ListView
from vertizee.classes.data_structures.vertex_dict import VertexDict
from vertizee.classes.vertex import V_co, VertexType

INFINITY: Final[float] = float("inf")


def reconstruct_path(
    source: "VertexType",
    destination: "VertexType",
    paths: Union["VertexDict[ShortestPath[V_co]]", "VertexDict[VertexDict[ShortestPath[V_co]]]"],
) -> List[V_co]:
    r"""Reconstructs the shortest path between two vertices based on the predecessors stored in the
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
        List[V_co]: The list of vertices comprising the path ``source`` :math:`\leadsto`
        ``destination``, or an empty list if no such path exists.
    """
    path_dict = cast(VertexDict[ShortestPath[V_co]], paths)
    if source in paths and isinstance(paths[source], VertexDict):
        path_dict = cast(VertexDict[ShortestPath[V_co]], paths[source])

    path = []
    v: Optional[VertexType] = destination
    while v is not None:
        path_v: Optional[ShortestPath[V_co]] = path_dict.get(v, None)
        if path_v is None:
            break
        if path_v.source != source:
            raise exception.AlgorithmError(
                "the ShortestPath object in the paths dictionary for "
                f"vertex '{v}' does not have source vertex '{source}'"
            )
        path.append(path_v.destination)
        v = path_v.predecessor

    if len(path) > 1:
        path = list(reversed(path))
    if path and path[0] != source:  # Check if a path exists from source to destination.
        path = []
    return path


class ShortestPath(Generic[V_co]):
    r"""Data structure representing a shortest path between a source vertex and a destination
    vertex.

    This class has a generic type parameter ``V``, which supports the type-hint usage
    ``ShortestPath[V_co]``.

    ``V = TypeVar("V", bound="VertexBase")`` See :class:`VertexBase
    <vertizee.classes.vertex.VertexBase>`.

    At a minimum, a shortest path includes:

        1) source - the starting vertex, designated as :math:`s`
        2) destination - the ending vertex, designated as :math:`d`
        3) predecessor - the vertex immediately preceding the destination vertex, designated as
           :math:`d.predecessor`
        4) length - the path length from source to destination, designated :math:`d.length`

    In addition, a shortest path may optionally include the list of vertices comprising the path.
    The default is to save memory space by not storing the path. When paths are not saved, they
    can be calculated as needed by calling :func:`reconstruct_path`.

    The convention in graph theory is to set the initial path length, :math:`d.length`, to
    infinity (:math:`\infty`), indicating that the destination is not reachable from the source.
    During the execution of a shortest-path algorithm, :math:`d.length` serves as an upper bound on
    a potential shortest path.

    Let :math:`\delta(s, d)` be the shortest-path distance from the source :math:`s` to
    destination :math:`d`.

    **Upper-bound property**:

        :math:`\forall d \in G(V),\ d.length \geq \delta(s, d)`, and once
        :math:`d.length` achieves the value :math:`\delta(s, d)` it never changes.

    **Operations that update a shortest path**:

        1) edge relaxation
        2) subpath relaxation

    Edge Relaxation
        The process of edge relaxation is testing whether one of the destination's incoming edges
        provides a shorter path from the source to the destination than the current estimate.

        Formally, let :math:`s` be the source vertex and :math:`(c, d)` be an incoming edge
        to the destination vertex :math:`d`. Define :math:`w` to be a weight function, such that
        :math:`w(u, v)` returns the weight (i.e. length) of some :math:`(u, v)` edge.

        | RELAX :math:`(c, d, w)`:
        | |emsp| |emsp| :math:`if\ d.length \gt c.length + w(c, d)`
        | |emsp| |emsp| |emsp| |emsp| :math:`d.length = c.length + w(c, d)`
        | |emsp| |emsp| |emsp| |emsp| :math:`d.predecessor = c`

    Subpath Relaxation
        Subpath relaxation is the process of testing whether some intermediate vertex :math:`k`
        yields two subpaths, that when combined, provide a shorter path from the source to the
        destination than the current estimate :math:`d.length`.

        Formally, let :math:`s \leadsto k` be a path connecting the source :math:`s` to some vertex
        :math:`k`, and let :math:`k \leadsto d` be a path from :math:`k` to the destination
        :math:`d`. Define the path length function :math:`\ell(x, y)` (script 'L') to be
        the function returning the length of some path :math:`x \leadsto y`.

        | RELAX-SUBPATHS :math:`(s \leadsto k,\ k \leadsto d)`:
        | |emsp| |emsp| :math:`if\ d.length \gt \ell(s,\ k) + \ell(k,\ d)`
        | |emsp| |emsp| |emsp| |emsp| :math:`d.length = \ell(s,\ k) + \ell(k,\ d)`
        | |emsp| |emsp| |emsp| |emsp| :math:`d.predecessor = predecessor(k \leadsto d)`

    Args:
        source: The source vertex of the path.
        destination: The destination vertex to which a path is to be found.
        initial_length: Optional; Sets the initial path length. Defaults to infinity.
        save_path: Optional; If True, all vertices in the shortest path are saved. Defaults to
            False.
    """

    def __init__(
        self,
        source: V_co,
        destination: V_co,
        initial_length: float = INFINITY,
        save_path: bool = False,
    ) -> None:
        if source is None:
            raise exception.VertizeeException("no source vertex specified")
        if destination is None:
            raise exception.VertizeeException("no destination vertex specified")
        self._source = source
        self._destination = destination

        self._edge_count: int = 0
        self._length: float = initial_length
        self._predecessor: Optional[V_co] = None
        self._store_full_path = save_path

        if self._store_full_path:
            self._path: Optional[List[V_co]] = [self.source]
        else:
            self._path = None

        self._add_s_d_edge_if_exists()

    def __len__(self) -> float:
        """Returns the length of this path from the source to the destination vertex. If there is no
        path connecting source to destination, then the length is infinity."""
        return self._length

    @property
    def destination(self) -> V_co:
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

    def path(self) -> "ListView[V_co]":
        """A :class:`ListView <vertizee.classes.collection_views.ListView>` of vertices comprising
        the path.

        If ``save_path`` was set to False upon initialization (the default), then the path list
        will always be empty. However, the path can be calculated using the function
        :func:`reconstruct_path`.

        See Also:
            * :func:`reconstruct_path`
        """
        if self._path is None or self._length == INFINITY or not self._store_full_path:
            return ListView([])
        return ListView(self._path)

    @property
    def predecessor(self) -> Optional[V_co]:
        """The vertex immediately preceding the destination vertex."""
        return self._predecessor

    def reinitialize(self, initial_length: float = INFINITY) -> None:
        """Reinitializes the shortest path by setting the initial length and clearing the
        ``path`` vertex list.

        Args:
            initial_length: Optional; Sets the initial path length. Defaults to infinity.
        """
        self._length = initial_length
        self._edge_count = 0

        if self._store_full_path:
            self._path = [self.source]

        self._add_s_d_edge_if_exists()

    def relax_edge(
        self,
        predecessor_path: "ShortestPath[V_co]",
        weight_function: Callable[[V_co, V_co, bool], Optional[float]],
        reverse_graph: bool = False,
    ) -> bool:
        """Tests whether there is a shorter path from the source vertex through ``predecessor_path``
        to this destination, and if so, updates this path to go through the predecessor path.

        This method returns False unless the following conditions are met:

         - This path and the predecessor path share the same source vertex.
         - There exists a :math:`(u, v)` edge, where :math:`u` is ``predecessor_path.destination``
           and :math:`v` is ``self.destination``, or if ``reverse_graph`` is True, there exists an
           edge :math:`(v, u)`.

        Args:
            predecessor_path: A graph path from the source to some vertex :math:`u`, such
                that there exists an edge :math:`(u, v)`, where :math:`v` is ``self.destination``.
            weight_function: A function that accepts two vertices and a boolean indicating if the
                graph is reversed (i.e. edges of directed graphs in the opposite direction) and
                returns the corresponding edge weight. If not provided, then the default
                ``Edge.weight`` property is used. For multigraphs, the lowest edge weight among the
                parallel edge connections is used.
            reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
                as if the graph were reversed (i.e. the :term:`reverse graph <reverse>`). Defaults
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

        j = predecessor_path.destination
        k = self.destination

        edge_len = weight_function(j, k, reverse_graph)
        if edge_len is None:  # This can happen if the weight function is designed as a filter.
            return False

        if self.length > predecessor_path.length + edge_len:
            self._edge_count += 1
            self._length = predecessor_path.length + edge_len
            self._predecessor = predecessor_path.destination

            if self._store_full_path:
                assert predecessor_path._path is not None
                self._path = predecessor_path._path.copy()
                self._path.append(self.destination)

            return True

        return False

    def relax_subpaths(
        self, path_s_k: "ShortestPath[V_co]", path_k_d: "ShortestPath[V_co]"
    ) -> bool:
        r"""Tests whether some vertex :math:`k` yields two subpaths, that when combined, provide a
        shorter path from the source :math:`s` to destination :math:`d` than the current estimate
        :math:`d.length`, and if so, updates the path by combing the two subpaths.

        This method returns False unless the following conditions are met:

         - The source of path :math:`s \leadsto k` is equal to ``self.source``.
         - The destination of path :math:`s \leadsto k` is equal to the source of path
           :math:`k \leadsto d`.

        Args:
            path_s_k: A path from the source to some vertex :math:`k`.
            path_k_d: A path from some vertex :math:`k` to the destination vertex.

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
            assert path_s_k._path is not None
            self._path = path_s_k._path.copy()
            if path_k_d.edge_count > 0:
                assert path_k_d._path is not None
                self._path.extend(path_k_d._path[1:])

        return True

    @property
    def source(self) -> V_co:
        """The source vertex of the path."""
        return self._source

    def _add_s_d_edge_if_exists(self) -> None:
        """If there exists an edge connecting source to destination, then the edge is added to the
        path and predecessor is set to source."""
        if self._source != self._destination:
            graph = self._source._parent_graph
            if graph.has_edge(self._source, self._destination):
                self._edge_count += 1
                self._predecessor = self._source
                if self._store_full_path:
                    assert self._path is not None
                    self._path.append(self._destination)
