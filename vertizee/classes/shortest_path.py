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

"""Data structure for working with shortest paths."""

from __future__ import annotations
from typing import Callable, List, Optional, TYPE_CHECKING, Union

# pylint: disable=cyclic-import
from vertizee import AlgorithmError
from vertizee.classes.collections.vertex_dict import VertexDict

if TYPE_CHECKING:
    from vertizee.classes.graph_base import GraphBase
    from vertizee.classes.vertex import Vertex, VertexKeyType

INFINITY = float("inf")


def reconstruct_path(
        source: VertexKeyType, destination: VertexKeyType,
        paths: Union[VertexDict['ShortestPath'], VertexDict[VertexDict['ShortestPath']]]
) -> List[Vertex]:
    """Reconstructs the path between two vertices based on the predecessors stored in the shortest
    paths dictionary (or a dictionary of shortest paths dictionaries).

    Args:
        source (Vertex): The first vertex in the path to be reconstructed.
        destination (Vertex): The last vertex in the path to be reconstructed.
        paths (Union[VertexDict['ShortestPath'], VertexDict[VertexDict['ShortestPath']]]): A
            dictionary of shortest paths (e.g. results from the single-source-shortest-paths
            problem) or a dictionary of dictionaries of shortest paths (e.g. results from the
            all-pairs-shortest-paths problem).

    Raises:
        AlgorithmError: an algorithm error is raised if one of the shortest path objects does not
            have a source vertex matching `source`.

    Returns:
        List[Vertex]: The list of vertices comprising the path source ~> destination, or an empty
        list if no such path exists.
    """
    path_dict = paths
    if source in paths:
        if isinstance(paths[source], VertexDict):
            path_dict = paths[source]

    path = []
    v = destination
    while v is not None:
        path_v: ShortestPath = path_dict.get(v, None)
        if path_v is None:
            break
        if path_v.source != source:
            raise AlgorithmError('the ShortestPath object in the paths dictionary for '
                                 f'vertex "{v}" does not have source vertex "{source}"')
        path.append(v)
        v = path_v.predecessor

    if len(path) > 1:
        path = list(reversed(path))
    if len(path) > 0 and path[0] != source:  # Check if a paths exists from source to destination.
        path = []
    return path


class ShortestPath:
    """Data structure representing a shortest path between some source vertex and a destination
    vertex.

    At a minimum, a shortest path is represented by:
        1) source vertex (s)
        2) destination vertex (d)
        3) predecessor (i.e. the vertex in the path immediately preceding the destination)
        4) length of the path from the source to destination (path_len(s, d))

    In addition, a shortest path may optionally include the list of vertices comprising a path
    from the source to the destination. The default is to save space and to instead call the
    function `reconstruct_path` as needed to calculate a given path based on the predecessor
    vertices.

    The convention in graph theory is to set the initial path length to infinity, indicating that
    the destination is not reachable from the source (that is, until/unless a path is found).
    During the execution of a shortest-path algorithm, the path length serves as an upper bound on
    a potential shortest path length.

    Let s be the source, d be the destination, path_len(s, d) be the current path length (i.e. the
    current upper bound of a shortest-path estimate), and w(u, v) be the weight/length of some
    arbitrary edge (u, v). Furthermore, define delta(s, d) as the shortest-path distance from
    s to d.

    The upper-bound property states:
        path_len(s, d) >= delta(s, d) for all vertices d in G(V), and once path_len(s, d) achieves
        the value delta(s, d), it never changes

    The two fundamental operations that update a shortest path are:
        1) edge relaxation
        2) subpath relaxation

    Edge Relaxation
        The process of edge relaxation is testing whether one of the destination's incoming edges
        provides a shorter path from the source to the destination than the current estimate.

        Formally, let (c, d) be one of the destination's incoming edges.

        relax_edge:
            if path_len(s, d) > path_len(s, c) + w(c, d):
                path_len(s, d) = path_len(s, c) + w(c, d)
                d.predecessor = c

    Subpath Relaxation
        Subpath relaxation is the process of testing whether some vertex k yields two subpaths,
        that when combined, provide a shorter path from the source to destination than the
        current estimate.

        Formally, let s ~> k be a path connecting the source to some vertex k, and let k ~> d be a
        path connecting k to the destination. Initially, the predecessors are defined as follows:

        Initialize predecessors:
            if s == d or w(s, d) == infinity:
                predecessor = None
            elif s != d and w(s, d) < infinity:
                predecessor = s

        relax_subpaths:
            if path_len(s, d) > path_len(s, k) + path_len(k, d):
                path_len(s, d) = path_len(s, k) + path_len(k, d)
                predecessor = predecessor(k, d)

    Args:
        source (Vertex): The source vertex of the path.
        destination (Vertex): The destination vertex to which a path is to be found.
        initial_length (float, optional): Sets the initial path length. Defaults to infinity.
        add_edge_to_destination_if_exists(bool, optional): If True, then if the initial
            length is less than infinity and there exists an edge connecting source to
            destination, the edge is added to the path. Defaults to True.
    """
    def __init__(self, source: Vertex, destination: Vertex, initial_length: float = INFINITY,
                 store_full_paths: bool = False, add_edge_to_destination_if_exists: bool = True):
        if source is None:
            raise ValueError('source was NoneType')
        if destination is None:
            raise ValueError('destination was NoneType')
        self._source: Vertex = source
        self._destination: Vertex = destination

        self._edge_count: int = 0
        self._length: float = initial_length
        self._predecessor: Vertex = None
        self._store_full_path = store_full_paths

        if self._store_full_path:
            self._path: List[Vertex] = [self.source]
        else:
            self._path: List[Vertex] = []

        if add_edge_to_destination_if_exists:
            self._init_edge_to_destination()

    @property
    def destination(self) -> Vertex:
        return self._destination

    @property
    def edge_count(self) -> int:
        return self._edge_count

    def is_destination_reachable(self) -> bool:
        """Returns True if the destination vertex is unreachable from the source vertex."""
        return self._length < INFINITY

    @property
    def length(self) -> float:
        """The length of this path from the source vertex to the destination vertex. If there is no
        path connecting source to destination, then the length should be infinity. The default
        value is infinity."""
        return self._length

    @property
    def path(self) -> List[Vertex]:
        """If the destination is reachable from the source, then the list of vertices of
        the path is returned. Note that if `store_full_paths` was set to False upon initialization,
        then the path returned will always be empty. See `~shortest_path.reconstruct_path`."""
        if self._length == INFINITY:
            return []
        else:
            return self._path.copy()

    @property
    def predecessor(self) -> Optional[Vertex]:
        return self._predecessor

    def reinitialize(self, initial_length: float = INFINITY,
                     add_edge_to_destination_if_exists: bool = True):
        """Reinitializes the shortest path by setting the initial length and clears intermediate
        vertices between the source and destination.

        Args:
            initial_length (float, optional): Sets the initial path length. Defaults to infinity.
            add_edge_to_destination_if_exists(bool, optional): If True, then if the initial
                length is less than infinity and there exists and edge connecting source to
                destination, then edge edge is added to the path. Defaults to True.
        """
        self._length: float = initial_length
        self._edge_count: int = 0

        if self._store_full_path:
            self._path: List[Vertex] = [self.source]
        else:
            self._path: List[Vertex] = []

        if add_edge_to_destination_if_exists:
            self._init_edge_to_destination()

    def relax_edge(self, predecessor_path: ShortestPath,
                   weight_function: Callable[[Vertex, Vertex, bool], float],
                   reverse_graph: bool = False) -> bool:
        """Test whether there is a shorter path from source through `predecessor_path` connected
        to this destination vertex via an edge (predecessor_path.destination, self.destination),
        and if so, update this path to go through the predecessor path.

        This method returns False unless the following conditions are met:
         - This path and the predecessor path share the same source vertex.
         - There exists an edge (predecessor_path.destination, self.destination), or if
               `reverse_graph` is True, there exists an edge
               (self.destination, predecessor_path.destination)

        For more details, see the class description for `~shortest_path.ShortestPath`.

        Args:
            predecessor_path (ShortestPath): A graph path from the source to some vertex, such
                that there exists and edge (predecessor_path.destination, self.destination).
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
            self._edge_count += 1
            self._length = predecessor_path.length + edge_len
            self._predecessor = predecessor_path.destination

            if self._store_full_path:
                self._path = predecessor_path.path
                self._path.append(self.destination)

            return True

        return False

    def relax_subpaths(self, path_s_k: ShortestPath, path_k_d: ShortestPath) -> bool:
        """Test whether some vertex k yields two subpaths, that when combined, provide a shorter
        path from the source to destination than the current estimate, and if so, update the path
        by combing the two subpaths.

        This method returns False unless the following conditions are met:
         - The source of path_s_k is equal to self._source.
         - The destination of path_s_k is equal to the source of path_k_d.

        For more details, see the class description for `~shortest_path.ShortestPath`.

        Args:
            path_s_k (ShortestPath): A graph path from the source to some vertex k.
            path_k_d (ShortestPath): A graph path from some vertex k to the destination vertex.

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
    def source(self) -> Vertex:
        return self._source

    def _init_edge_to_destination(self):
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
