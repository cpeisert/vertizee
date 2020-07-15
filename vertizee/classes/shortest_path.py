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

"""Data structure for working with a path that may be a shortest path between some source vertex
and destination vertex."""

from __future__ import annotations
from typing import Callable, List, Optional, TYPE_CHECKING

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.graph_base import GraphBase
    from vertizee.classes.vertex import Vertex

INFINITY = float("inf")
NEG_INFINITY = float("-inf")


class ShortestPath:
    """Data structure representing a shortest path between some source vertex and a destination
    vertex.

    This data structure stores paths as a sequence of vertices. The default initialization state
    sets the path length to infinity. If at the end of a graph search the path length is infinity,
    then the destination should not be reachable from the source (depending on the algorithm).

    Args:
        source (Vertex): The source vertex of the path.
        destination (Vertex): The destination vertex to which a path is to be found.
        initial_length (float, optional): Sets the initial path length. Defaults to infinity.
    """
    def __init__(self, source: Vertex, destination: Vertex, initial_length: float = INFINITY):
        if source is None:
            raise ValueError('source was NoneType')
        if destination is None:
            raise ValueError('destination was NoneType')
        self._source: Vertex = source
        self._destination: Vertex = destination

        self._edge_count: int = 0
        self._length: float = initial_length
        self._path: List[Vertex] = [self.source]
        self._path_contains_destination: bool = False

    def add_edge(self, adj_vertex: Vertex, edge_length: float = 1):
        """Adds a new edge by specifying a vertex adjacent to the last vertex in the current path.

        Args:
            adj_vertex (Vertex): The vertex to add to the path.
            edge_length (float, optional): The length/weight of the edge being added. Defaults to
                1.
        """
        self._edge_count += 1
        self._length += edge_length
        self._path.append(adj_vertex)
        if adj_vertex == self._destination:
            self._path_contains_destination = True

    def clone_from_excluding_destination(self, shortest_path: 'ShortestPath'):
        """Clone the state of `shortest_path` to this object, but exclude the destination vertex,
        which remains unchanged after object initialization.

        Args:
            shortest_path (ShortestPath): The ShortestPath to clone.
        """
        self._source = shortest_path._source
        self._edge_count = shortest_path._edge_count
        self._path = shortest_path._path.copy()
        self._length = shortest_path._length

    @property
    def destination(self) -> Vertex:
        return self._destination

    @property
    def edge_count(self) -> int:
        return self._edge_count

    def is_destination_unreachable(self) -> bool:
        """Returns True if the destination vertex is unreachable from the source vertex."""
        return self._length == INFINITY or self._length == NEG_INFINITY

    @property
    def length(self) -> float:
        """The length of this path from the source vertex to the destination vertex. If there is no
        path connecting source to destination, then the length should be infinity or negative
        infinity. The default value is infinity."""
        return self._length

    @property
    def path(self) -> List[Vertex]:
        return self._path.copy()

    def path_contains_destination(self) -> bool:
        """Returns True if the path contains the destination vertex."""
        return self._path_contains_destination

    def reinitialize(self, initial_length: float = INFINITY):
        """Reinitializes the shortest path by setting the initial length and clears intermediate
        vertices between the source and destination.

        Args:
            initial_length (float, optional): Sets the initial path length. Defaults to infinity.
        """
        self._length: float = initial_length

        self._edge_count: int = 0
        self._path: List[Vertex] = [self._source]

    def relax(self, predecessor_path: 'ShortestPath',
              weight_function: Optional[Callable[[Vertex, Vertex, bool], float]],
              reverse_graph: bool = False) -> bool:
        """If there is a shorter path to this path's destination vertex passing through some
        predecessor vertex (i.e. `predecessor_path.destination`), then update this path to go
        through the predecessor.

        Formally, let the predecessor path be path_j defined as the sequence of vertices
        [s, x1, x2, ..., j] with source 's' and destination 'j'. Let this (self) path be
        path_k with vertices [s, y1, y2,..., k]. If there exists an edge (j, k) with weight
        w(j, k), then:

        if path_k.length > path_j.length + w(j, k)
            path_k.length = path_j.length + w(j, k)
            path_k.path = [s, x1, x2,..., j, k]

        This method returns False unless the following conditions are met:
         - This path and the predecessor path share the same source vertex.
         - There exists an edge (predecessor_path.destination, self.destination), or if
               `reverse_graph` is True, there exists an edge
               (self.destination, predecessor_path.destination)

        Args:
            predecessor_path (ShortestPath): A graph path from the source to some vertex u that has
                an edge connecting u to the destination of this path.
            weight_function (Callable[[Vertex, Vertex, bool], float]): A function that accepts
                two vertices and a boolean indicating if the graph is reversed (i.e. edges of
                directed graphs in the opposite direction) and returns the corresponding edge
                weight. If not provided, then the default `Edge.weight` property is used. For
                multigraphs, the lowest edge weight among the parallel edges is used.
            reverse_graph (bool, optional): For directed graphs, setting to True will yield a
                traversal as if the graph were reversed (i.e. the reverse/transpose/converse
                graph). Defaults to False.

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
            self._length = predecessor_path.length
            self._path = predecessor_path.path
            self.add_edge(self.destination, edge_length=edge_len)
            return True

        return False

    @property
    def source(self) -> Vertex:
        return self._source
