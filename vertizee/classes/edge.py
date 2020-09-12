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

"""Data types supporting directed and undirected graph edges.

* Edge - Class supporting both directed and undirected edges.
* DiEdge - A directed edge class with properties for the tail and head vertices.
* EdgeType - A union of the edge classes for generic type hinting.
"""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING, Union

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.vertex import Vertex

EdgeType = Union["DiEdge", "Edge"]

DEFAULT_WEIGHT = 1


class Edge:
    """Edge is a graph primitive that represents an undirected connection between two vertices.

    To ensure the integrity of graphs, edges should never be initialized directly. Attempting
    to initialize an edge using its `__init__` method will raise an error. To create edges, use
    GraphBase methods such as `add_edge`.

    Edges may be assigned a weight as well as custom attributes. This class supports both
    directed and undirected edges. However, directed graphs use the subclass DiEdge, which provides
    tail and head Vertex properties.

    IMPORTANT: If multiple edges are added with the same vertices, then a single `Edge` instance
    is used to store the parallel edges. When working with edges, use the `parallel_edge_count`
    property to determine if the edge represents more than one edge connection. Edges represented
    by vertex pairs (a, b) and (b, a) map to the same Edge object in undirected graphs, but
    different Edge objects in directed graphs.

    Args:
        v1: The first vertex (the order does not matter, since this is an
            undirected edge).
        v2: The second vertex.
        weight: Optional; Edge weight. Defaults to 1.
        parallel_edge_count: Optional; The number of parallel edges from vertex1 to vertex2.
            Defaults to 0.
        parallel_edge_weights: Optional; List of weights for the parallel edges.
            Defaults to None.

    Note:
        When new edges are created, the incident edge lists for each Vertex object are updated
        with references to the new edge. This process is handled automatically as long as vertices
        and edges are not initialized outside of the `GraphBase` API.
    """

    # Limit initialization to protected method `_create`.
    _create_key = object()

    @classmethod
    def _create(
        cls,
        v1: Vertex,
        v2: Vertex,
        weight: Optional[float] = DEFAULT_WEIGHT,
        parallel_edge_count: Optional[int] = 0,
        parallel_edge_weights: Optional[List[float]] = None,
    ):
        """Initializes a new Edge."""
        return Edge(
            cls._create_key,
            v1=v1,
            v2=v2,
            weight=weight,
            parallel_edge_count=parallel_edge_count,
            parallel_edge_weights=parallel_edge_weights,
        )

    def __init__(
        self,
        create_key,
        v1: Vertex,
        v2: Vertex,
        weight: Optional[float] = DEFAULT_WEIGHT,
        parallel_edge_count: Optional[int] = 0,
        parallel_edge_weights: Optional[List[float]] = None,
    ):
        if create_key != Edge._create_key:
            raise ValueError(
                f"{self._runtime_type()} objects must be initialized using " "`_create`."
            )

        # IMPORTANT: vertex1 and vertex2 are used in Edge.__hash__, and must therefore be
        # treated as immutable (read-only). If the vertex keys need to change, first delete the
        # edge instance and then create a new instance.
        self._vertex1 = v1
        self._vertex2 = v2

        self.attr: dict = {}
        """Custom attribute dictionary to store any additional data associated with edges."""

        self._weight: float = float(weight)

        self._parallel_edge_count = parallel_edge_count
        if parallel_edge_weights is None:
            self._parallel_edge_weights: List[float] = []
        else:
            self._parallel_edge_weights: List[float] = [float(x) for x in parallel_edge_weights]

        self._parent_graph: "GraphBase" = self.vertex1._parent_graph

        # Don't raise warning, unless edge has non-default weight.
        if self._weight != DEFAULT_WEIGHT and self._parallel_edge_count != len(
            self._parallel_edge_weights
        ):
            raise RuntimeWarning(
                f"The parallel edge count ({self._parallel_edge_count})"
                f" is not equal to the number of parallel edge weight entries "
                f"({len(self._parallel_edge_weights)})."
            )

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False

        directed_graph = self.vertex1._parent_graph.is_directed_graph()
        v1 = self.vertex1
        v2 = self.vertex2
        o_v1 = other.vertex1
        o_v2 = other.vertex2
        if not directed_graph:
            if v1.key > v2.key:
                v1, v2 = v2, v1
            if o_v1.key > o_v2.key:
                o_v1, o_v2 = o_v2, o_v1
        if (
            v1 != o_v1
            or v2 != o_v2
            or self._parallel_edge_count != other._parallel_edge_count
            or self._weight != other._weight
        ):
            return False
        return True

    def __hash__(self):
        """Create a hash key using the edge vertices.

        Note that __eq__ is defined to take `_weight`, and `_parallel_edge_count` into
        consideration, whereas `__hash__` does not. This is because `_weight` and
        `_parallel_ege_count` are not intended to be immutable throughout the lifetime of the
        object.

        From: "Python Hashes and Equality" <https://hynek.me/articles/hashes-and-equality/>:

        "Hashes can be less picky than equality checks. Since key lookups are always followed by
        an equality check, your hashes don’t have to be unique. That means that you can compute
        your hash over an immutable subset of attributes that may or may not be a unique
        'primary key' for the instance."
        """
        directed_graph = self.vertex1._parent_graph.is_directed_graph()
        if directed_graph:
            return hash((self.vertex1, self.vertex2))
        else:  # undirected graph
            if self.vertex1.key <= self.vertex2.key:
                return hash((self.vertex1, self.vertex2))
            else:
                return hash((self.vertex2, self.vertex1))

    def __str__(self):
        directed_graph = self.vertex1._parent_graph.is_directed_graph()
        if directed_graph:
            return f"({self.vertex1.key}, {self.vertex2.key})"
        else:  # Undirected edge
            if self.vertex1.key <= self.vertex2.key:
                return f"({self.vertex1.key}, {self.vertex2.key})"
            else:
                return f"({self.vertex2.key}, {self.vertex1.key})"

    def is_loop(self) -> bool:
        """Returns True if this edge is a loop back to itself."""
        return self.vertex1.key == self.vertex2.key

    @property
    def parallel_edge_count(self) -> int:
        """The number of parallel edges.

        Note:
            The parallel edge count will always be one less than the total number of edges, since
            the initial edge connecting the two endpoints is not counted as a parallel edge.

        Returns:
            int: The number of parallel edges.
        """
        return self._parallel_edge_count

    @property
    def parallel_edge_weights(self) -> List[float]:
        """A list of parallel edge weights.

        This list should not contain the weight of the initial edge between the two vertices,
        which should instead be stored in `_weight`.

        Returns:
            List[float]: The list of parallel edge weights.
        """
        return self._parallel_edge_weights.copy()

    @property
    def vertex1(self) -> Vertex:
        return self._vertex1

    @property
    def vertex2(self) -> Vertex:
        return self._vertex2

    @property
    def weight(self) -> float:
        """Returns the edge weight."""
        return self._weight

    @property
    def weight_with_parallel_edges(self) -> float:
        """The total weight, including parallel edges.

        Note:
            For directed graphs, this method returns the total weight of all edges between
            (vertex1, vertex2), but it does not include the weights of directed edges from
            (vertex2, vertex1).

        Returns:
            float: The total edge weight, including parallel edges.
        """
        total = self._weight
        for w in self._parallel_edge_weights:
            total += w
        return total

    def _runtime_type(self):
        """Returns the name of the runtime subclass."""
        return self.__class__.__name__


class DiEdge(Edge):
    """DiEdge is a graph primitive that represents a directed connection between two vertices.

    To ensure the integrity of graphs, edges should never be initialized directly. Attempting
    to initialize an edge using its `__init__` method will raise an error. To create edges, use
    GraphBase methods such as `add_edge`.

    The starting vertex is called the tail and the destination vertex is called the head. The edge
    points from the tail to the head.

    Args:
        vertex1: The first vertex.
        vertex2: The second vertex.
        weight: Optional; Edge weight. Defaults to 0.0.
        parallel_edge_count: Optional; The number of parallel edges from vertex1 to vertex2.
            Defaults to 0.
        parallel_edge_weights: Optional; List of weights for the parallel edges.
            Defaults to None.
    """

    # Limit initialization to protected method `_create`.
    __create_key = object()

    # pylint: disable=arguments-differ
    @classmethod
    def _create(
        cls,
        tail: Vertex,
        head: Vertex,
        weight: Optional[float] = DEFAULT_WEIGHT,
        parallel_edge_count: Optional[int] = 0,
        parallel_edge_weights: Optional[List[float]] = None,
    ) -> "DiEdge":
        """Initializes a new Edge."""
        return DiEdge(
            cls.__create_key,
            tail=tail,
            head=head,
            weight=weight,
            parallel_edge_count=parallel_edge_count,
            parallel_edge_weights=parallel_edge_weights,
        )

    def __init__(
        self,
        create_key,
        tail: Vertex,
        head: Vertex,
        weight: Optional[float] = DEFAULT_WEIGHT,
        parallel_edge_count: Optional[int] = 0,
        parallel_edge_weights: Optional[List[float]] = None,
    ):
        if create_key != DiEdge.__create_key:
            raise ValueError(
                f"{self._runtime_type()} objects must be initialized using " "`_create`."
            )
        super().__init__(
            Edge._create_key, tail, head, weight, parallel_edge_count, parallel_edge_weights
        )

    @property
    def head(self) -> Vertex:
        return self._vertex2

    @property
    def tail(self) -> Vertex:
        return self._vertex1
