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

* :class:`Edge <vertizee.classes.edge.Edge>` - An undirected connection between two vertices.
  The order of the vertices does not matter. However, the string representation will always
  print vertices sorted in lexicographic order.
* :class:`DiEdge <vertizee.classes.edge.DiEdge>` - A directed connection between two vertices that
  defines the ``tail`` as the starting vertex and the ``head`` as the destination vertex.
* :class:`EdgeType` - A type alias for the union of the edge classes: ``Union[DiEdge, Edge]``
"""

from __future__ import annotations
from typing import Any, List, Optional, TYPE_CHECKING, Union

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.vertex import Vertex

EdgeType = Union["DiEdge", "Edge"]

DEFAULT_WEIGHT = 1


class Edge:
    """Edge is a graph primitive that represents an undirected connection between two vertices.

    To ensure the integrity of graphs, edges should never be initialized directly. Attempting
    to initialize an edge using its ``__init__`` method will raise an error. To create edges, use
    :class:`GraphBase <vertizee.classes.graph_base.GraphBase>` methods such as
    :meth:`GraphBase.add_edge <vertizee.classes.graph_base.GraphBase.add_edge>`.

    Edges may be assigned a weight as well as custom attributes using the :attr:`attr` dictionary.

    Note:
        Directed graphs use the subclass :class:`DiEdge`, which provides ``tail`` and ``head``
        Vertex properties.

    Note:
        If multiple edges are added with the same vertices, then a single ``Edge`` instance
        is used to store the parallel edges. When working with edges, use the
        ``parallel_edge_count`` property to determine if the edge represents more than one edge
        connection. Edges represented by vertex pairs :math:`(a, b)` and :math:`(b, a)` map to the
        same ``Edge`` object in undirected graphs, but different ``DiEdge`` objects in directed
        graphs.

    Note:
        When new edges are created, the incident edge lists for each ``Vertex`` object are updated
        with references to the new edge. This process is handled automatically as long as vertices
        and edges are initialized using the ``GraphBase`` API (e.g. :meth:`GraphBase.add_edge
        <vertizee.classes.graph_base.GraphBase.add_edge>` and :meth:`GraphBase.add_vertex
        <vertizee.classes.graph_base.GraphBase.add_vertex>`).

    Args:
        v1: The first vertex (the order does not matter, since this is an
            undirected edge).
        v2: The second vertex.
        weight: Optional; Edge weight. Defaults to 1.
        parallel_edge_count: Optional; The number of parallel edges from vertex1 to vertex2.
            Defaults to 0.
        parallel_edge_weights: Optional; List of weights for the parallel edges.
            Defaults to None.

    Attributes:
        attr: Attribute dictionary to store ad hoc data associated with the edge.
    """

    # Limit initialization to protected method `_create`.
    _create_key = object()

    @classmethod
    def _create(
        cls,
        v1: "Vertex",
        v2: "Vertex",
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
        v1: "Vertex",
        v2: "Vertex",
        weight: float = DEFAULT_WEIGHT,
        parallel_edge_count: int = 0,
        parallel_edge_weights: Optional[List[float]] = None,
    ):
        if create_key != Edge._create_key:
            raise ValueError(
                f"{self._runtime_type()} objects should be created using method "
                "GraphBase.add_edge(); do not use __init__"
            )

        # IMPORTANT: _vertex1 and _vertex2 are used in Edge.__hash__, and must therefore be
        # treated as immutable (read-only). If the vertex keys need to change, first delete the
        # edge instance and then create a new instance.
        self._vertex1 = v1
        self._vertex2 = v2

        self.attr: dict = {}
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
            if v1.label > v2.label:
                v1, v2 = v2, v1
            if o_v1.label > o_v2.label:
                o_v1, o_v2 = o_v2, o_v1
        if (
            v1 != o_v1
            or v2 != o_v2
            or self._parallel_edge_count != other._parallel_edge_count
            or self._weight != other._weight
        ):
            return False
        return True

    def __getitem__(self, key: Any) -> Any:
        """Supports index accessor notation to retrieve values from the `attr` dictionary.

        Example:
            >>> import vertizee as vz
            >>> g = vz.Graph()
            >>> g.add_edge(1, 2)
            (1, 2)
            >>> g[1, 2]["color"] = "blue"
            >>> g[1, 2]["color"]
            'blue'

        Args:
            key: The `attr` dictionary key.

        Returns:
            Any: The value indexed by `key`.
        """
        return self.attr[key]

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

        # undirected graph
        if self.vertex1.label > self.vertex2.label:
            return hash((self.vertex2, self.vertex1))

        return hash((self.vertex1, self.vertex2))

    def __repr__(self):
        return self.__str__()

    def __setitem__(self, key: Any, value: Any):
        """Supports index accessor notation to set values in the `attr` dictionary.

        Example:
            >>> import vertizee as vz
            >>> g = vz.Graph()
            >>> g.add_edge(1, 2)
            (1, 2)
            >>> g[1, 2]["color"] = "blue"
            >>> g[1, 2]["color"]
            "blue"

        Args:
            key: The `attr` dictionary key.
            value: The value to assign to `key` in the `attr` dictionary.
        """
        self.attr[key] = value

    def __str__(self):
        directed_graph = self.vertex1._parent_graph.is_directed_graph()
        if directed_graph:
            edge_str = f"({self.vertex1.label}, {self.vertex2.label}"
        else:
            # undirected edge
            if self.vertex1.label > self.vertex2.label:
                edge_str = f"({self.vertex2.label}, {self.vertex1.label}"
            else:
                edge_str = f"({self.vertex1.label}, {self.vertex2.label}"

        if self.vertex1._parent_graph.is_weighted():
            edge_str = f"{edge_str}, {self._weight})"
        else:
            edge_str = f"{edge_str})"

        edges = [edge_str for _ in range(self.parallel_edge_count + 1)]
        return ", ".join(edges)

    def is_loop(self) -> bool:
        """Returns True if this edge is a loop back to itself."""
        return self.vertex1.label == self.vertex2.label

    @property
    def multiplicity(self) -> int:
        """The multiplicity is the number of edges within a multi-edge.

        For edges without parallel connections, the multiplicity is 1. Each parallel edge adds 1 to
        the multiplicity. A edge with one parallel connection has multiplicity 2.
        """
        return self._parallel_edge_count + 1

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

        The list of parallel edge weights does not contain the weight of the initial edge between
        the two vertices. Instead, the initial edge weight is stored in the ``weight`` property.

        Returns:
            List[float]: The list of parallel edge weights.
        """
        return self._parallel_edge_weights.copy()

    @property
    def vertex1(self) -> Vertex:
        """The first vertex. For DiEdge objects, this is a synonym for the ``tail`` property."""
        return self._vertex1

    @property
    def vertex2(self) -> Vertex:
        """The second vertex. For DiEdge objects, this is a synonym for the ``head`` property."""
        return self._vertex2

    @property
    def weight(self) -> float:
        """The edge weight."""
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
    to initialize an edge using its ``__init__`` method will raise an error. To create edges, use
    :class:`GraphBase <vertizee.classes.graph_base.GraphBase>` methods such as
    :meth:`GraphBase.add_edge <vertizee.classes.graph_base.GraphBase.add_edge>`.

    The starting vertex is called the ``tail`` and the destination vertex is called the ``head``.
    The edge points from the tail to the head.

    DiEdges may be assigned a weight as well as custom attributes using the :attr:`attr` dictionary.

    Args:
        tail: The starting vertex.
        head: The destination vertex.
        weight: Optional; Edge weight. Defaults to 0.0.
        parallel_edge_count: Optional; The number of parallel edges from tail to head.
            Defaults to 0.
        parallel_edge_weights: Optional; List of weights for the parallel edges.
            Defaults to None.

    Attributes:
        attr: Attribute dictionary to store ad hoc data associated with the edge.
    """

    # Limit initialization to protected method `_create`.
    __create_key = object()

    # pylint: disable=arguments-differ
    @classmethod
    def _create(
        cls,
        tail: Vertex,
        head: Vertex,
        weight: float = DEFAULT_WEIGHT,
        parallel_edge_count: int = 0,
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
        weight: float = DEFAULT_WEIGHT,
        parallel_edge_count: int = 0,
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
        """The destination vertex that the tail points to. This is a synonym for the ``vertex2``
        property."""
        return self._vertex2

    @property
    def tail(self) -> Vertex:
        """The starting vertex that points to the head. This is a synonym for the ``vertex1``
        property."""
        return self._vertex1
