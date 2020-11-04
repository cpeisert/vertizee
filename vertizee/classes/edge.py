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

Classes and type aliases:

* :class:`Connection[V] <Connection>` - A single connection between two vertices in a graph.
* :class:`MultiConnection[V] <MultiConnection>` - A connection that may include multiple parallel
  connections between two vertices in a graph.
* :class:`EdgeViewBase[V] <EdgeViewBase>` - A dynamic view of an edge. Edge views provide an
  edge-like API for each of the parallel edge connections in a multiedge.
* :class:`EdgeView(EdgeViewBase[Vertex]) <EdgeView>` - A dynamic view of an undirected edge. Edge
  views provide an edge-like API for each of the parallel edge connections in a multiedge.
* :class:`DiEdgeView(EdgeViewBase[DiVertex]) <DiEdgeView>` - A dynamic view of a directed edge.
  Edge views provide an edge-like API for each of the parallel edge connections in a multiedge.
* :class:`EdgeBase(Connection[V]) <EdgeBase>` - Abstract base class from which all single-connection
  edge classes inherit.
* :class:`MultiEdgeBase(MultiConnection[V]) <MultiEdgeBase>` - Abstract base class from which all
  multiedge classes inherit.
* :class:`Edge(EdgeBase[Vertex]) <Edge>` - An undirected edge that does not allow parallel
  connections between its vertices. This is the class to use when adding type hints for undirected
  edge objects.
* :class:`DiEdge(EdgeBase[DiVertex]) <DiEdge>` - A directed connection between two vertices that
  defines the ``tail`` as the starting vertex and the ``head`` as the destination vertex. Parallel
  connections are not allowed. This is the class to use when adding type hints for directed edge
  objects.
* :class:`MultiEdge(MultiEdgeBase[Vertex]) <MultiEdge>` - An undirected multiedge that allows
  multiple parallel connections between its two vertices. This is the class to use when adding type
  hints for undirected multiedge objects.
* :class:`MultiDiEdge(MultiEdgeBase[DiVertex]) <MultiDiEdge>` - A directed multiedge that allows
  multiple directed connections between its two vertices. The ``tail`` is the starting vertex and
  the ``head`` is the destination vertex. This is the class to use when adding type hints for
  directed multiedge objects.
* :class:`EdgeType` - A type alias defined as
  ``Union[DiEdge, Edge, EdgeLiteral, MultiDiEdge, MultiEdge]`` where ``EdgeLiteral`` is an alias
  for various edge-tuple formats, such as ``Tuple[VertexType, VertexType]`` and
  ``Tuple[VertexType, VertexType, Weight]``.
* :class:`V` - A generic type parameter that may be either a :class:`Vertex
  <vertizee.classes.vertex.Vertex>` or :class:`DiVertex <vertizee.classes.vertex.DiVertex>`.

Functions:

* :func:`create_edge_label` - Creates a consistent string representation of the edge. Directed
  edges have labels with the vertices ordered based on the initialization order. Undirected edges
  have labels with the vertices sorted lexicographically. For example, both :math:`(1, 2)` and
  :math:`(2, 1)` refer to the same undirected edge connection, but the label would always be
  "(1, 2)".
"""
# pylint: disable=unsubscriptable-object
# See pylint issue #2822: https://github.com/PyCQA/pylint/issues/2822

from __future__ import annotations
from abc import ABC, abstractmethod
import numbers
from typing import (
    Any, Dict, Final, Generic, Hashable, Iterable, Iterator, List, Optional, Tuple, TYPE_CHECKING,
    TypeVar, Union
)

from vertizee.classes import vertex as vertex_module
from vertizee.classes.vertex import DiVertex, MultiDiVertex, MultiVertex, Vertex, VertexType
from vertizee.utils import abc_utils

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.graph import GraphBase

# Type aliases
AttributesDict = dict
Weight = float
EdgeClass = Union["DiEdge", "Edge", "MultiDiEdge", "MultiEdge"]
EdgeTuple = Tuple[VertexType, VertexType]
EdgeTupleWeighted = Tuple[VertexType, VertexType, Weight]
EdgeTupleAttr = Tuple[VertexType, VertexType, AttributesDict]
EdgeTupleWeightedAttr = Tuple[VertexType, VertexType, Weight, AttributesDict]
EdgeLiteral = Union[EdgeTuple, EdgeTupleWeighted, EdgeTupleAttr, EdgeTupleWeightedAttr]

#: EdgeType: A type alias defined as ``Union[EdgeClass, EdgeLiteral]``,
# where ``EdgeLiteral`` is an alias for various edge-tuple formats.
EdgeType = Union[EdgeClass, EdgeLiteral]

ConnectionKey = Hashable

#: V: A generic type parameter that represents a :class:`Vertex <vertizee.classes.vertex.Vertex>`
# or :class:`DiVertex <vertizee.classes.vertex.DiVertex>`.
V = TypeVar("V", DiVertex, Vertex)

DEFAULT_WEIGHT: Final = 1.0
DEFAULT_CONNECTION_KEY: Final = 0


def create_edge_label(vertex1: "VertexType", vertex2: "VertexType", is_directed: bool) -> str:
    """Creates a consistent string representation of the edge.

    Directed edges have labels with the vertices ordered based on the initialization order.
    Undirected edges have labels with the vertices sorted lexicographically. For example, both
    :math:`(1, 2)` and  :math:`(2, 1)` refer to the same undirected edge connection, but the label
    would always be "(1, 2)".

    Args:
        vertex1: The first vertex of the edge.
        vertex2: The second vertex of the edge.
        is_directed (bool): True indicates a directed edge, False an undirected edge.

    Returns:
        str: The edge label.
    """
    v1_label = vertex_module.get_vertex_label(vertex1)
    v2_label = vertex_module.get_vertex_label(vertex2)

    if not is_directed and v1_label > v2_label:
        return f"({v2_label}, {v1_label})"
    return f"({v1_label}, {v2_label})"


def _are_connections_equal(connection: "Connection", other: "Connection") -> bool:
    """Returns ``True`` if ``connection`` is logically equal to ``other``."""
    if connection is other:
        return True
    v1 = connection.vertex1
    v2 = connection.vertex2
    o_v1 = other.vertex1
    o_v2 = other.vertex2

    if not v1._parent_graph.is_directed():
        if v1.label > v2.label:
            v1, v2 = v2, v1
        if o_v1.label > o_v2.label:
            o_v1, o_v2 = o_v2, o_v1

    if v1 != o_v1 or v2 != o_v2 or connection.weight != other.weight:
        return False
    if connection.has_attributes_dict() != other.has_attributes_dict():
        return False
    if connection.has_attributes_dict() and other.has_attributes_dict():
        if connection.attr != other.attr:
            return False
    return True


def _are_multiconnections_equal(
    multiconnection: "MultiConnection", other: "MultiConnection"
) -> bool:
    """Returns ``True`` if ``multiconnection`` is logically equal to ``other``. Note that
    multiconnection equality does not rely on the attributes of individual parallel connections
    within multiconnections."""
    if multiconnection is other:
        return True
    v1 = multiconnection.vertex1
    v2 = multiconnection.vertex2
    o_v1 = other.vertex1
    o_v2 = other.vertex2

    if not v1._parent_graph.is_directed():
        if v1.label > v2.label:
            v1, v2 = v2, v1
        if o_v1.label > o_v2.label:
            o_v1, o_v2 = o_v2, o_v1

    if v1 != o_v1 or v2 != o_v2 or multiconnection.weight != other.weight:
        return False
    if multiconnection.multiplicity != other.multiplicity:
        return False
    return True


def _contract_connection(edge: Connection, remove_loops: bool = False) -> None:
    """Contracts an edge by removing it from the graph and merging its two incident vertices.

    Args:
        edge: The edge to contract.
        remove_loops: If True, loops on the merged vertex will be removed. Defaults to False.
    """
    graph = edge._parent_graph
    v1 = edge.vertex1
    v2 = edge.vertex2

    edges_to_delete: List[EdgeClass] = []
    # Incident edges of vertex2, where vertex2 is to be replaced by vertex1.
    for incident in v2.incident_edges:
        edges_to_delete.append(incident)

        if incident.is_loop():
            vertex1 = v1
            vertex2 = v1
        elif incident.vertex1 == v2:
            vertex1 = v1
            vertex2 = incident.vertex2
        else:  # incident.vertex2 == v2
            vertex1 = incident.vertex1
            vertex2 = v1

        if not graph.has_edge(vertex1, vertex2):
            graph.add_edge(vertex1, vertex2, weight=incident.weight, **incident.attr)

    # Delete indicated edges after finishing loop iteration.
    for e in edges_to_delete:
        e.remove()
    if remove_loops or not graph.allows_self_loops():
        v1.remove_loops()
    graph.remove_vertex(v2)


def _contract_multiconnection(edge: MultiConnection, remove_loops: bool = False) -> None:
    """Contracts a multiedge by removing it from the graph and merging its two incident vertices.

    Args:
        edge: The edge to contract.
        remove_loops: If True, loops on the merged vertex will be removed. Defaults to False.
    """
    graph = edge._parent_graph
    v1 = edge.vertex1
    v2 = edge.vertex2

    edges_to_delete: List[EdgeClass] = []
    # Incident edges of vertex2, where vertex2 is to be replaced by vertex1.
    for incident in v2.incident_edges:
        edges_to_delete.append(incident)

        if incident.is_loop():
            vertex1 = v1
            vertex2 = v1
        elif incident.vertex1 == v2:
            vertex1 = v1
            vertex2 = incident.vertex2
        else:  # incident.vertex2 == v2
            vertex1 = incident.vertex1
            vertex2 = v1

        existing_keys = set()
        if graph.has_edge(vertex1, vertex2):
            for key, _ in graph[vertex1, vertex2].connection_items():
                existing_keys.add(key)

        for key, connection in incident.connection_items():

            if vertex1 == vertex2:
                print(f"DEBUG: Adding loop connection {vertex1, vertex2}")

            if key in existing_keys:
                graph.add_edge(vertex1, vertex2, weight=connection.weight)
            else:
                graph.add_edge(vertex1, vertex2, weight=connection.weight, key=key)

    # Delete indicated edges after finishing loop iteration.
    for e in edges_to_delete:
        e.remove()
    if remove_loops or not graph.allows_self_loops():
        v1.remove_loops()
    graph.remove_vertex(v2)


def _create_connection_key(connection_keys: Iterable[ConnectionKey]):
    """Creates a new connection key that does not conflict with a collection of existing keys."""
    numeric_keys = [int(k) for k in connection_keys if isinstance(k, numbers.Number)]
    if numeric_keys:
        max_key = max(numeric_keys)
        new_key = max_key + 1
    else:
        new_key = DEFAULT_CONNECTION_KEY + 1
    return new_key


def _str_for_connection(vertex1: V, vertex2: V, weight: float) -> str:
    """A simple string representation of the edge connection that shows the vertex labels, and for
    weighted graphs, includes the edge weight."""
    edge_str = f"({vertex1.label}, {vertex2.label}"

    if not vertex1._parent_graph.is_directed():
        if vertex1.label > vertex2.label:
            edge_str = f"({vertex2.label}, {vertex1.label}"

    if vertex1._parent_graph.is_weighted():
        edge_str = f"{edge_str}, {weight})"
    else:
        edge_str = f"{edge_str})"

    return edge_str


def _str_for_multiconnection(
    vertex1: V, vertex2: V, connections: Dict[ConnectionKey, _EdgeConnectionData]
) -> str:
    """A simple string representation of the multiedge that shows the vertex labels, and for
    weighted graphs, includes the edge weight. The string will show separate vertex tuples for each
    parallel connection, such as "(a, b, 1.0), (a, b, 3.5), (a, b, 2.1)"."""
    connection_str = f"({vertex1.label}, {vertex2.label}"

    if not vertex1._parent_graph.is_directed():
        if vertex1.label > vertex2.label:
            connection_str = f"({vertex2.label}, {vertex1.label}"

    multiconnection_strings = list()
    if vertex1._parent_graph.is_weighted():
        for connection in connections.values():
            multiconnection_strings.append(f"{connection_str}, {connection.weight})")
    else:
        connection_str = f"{connection_str})"
        multiconnection_strings = [connection_str] * len(connections)

    return ", ".join(multiconnection_strings)


class Connection(ABC, Generic[V]):
    """A single connection between two vertices in a graph."""

    __slots__ = ()

    # mypy: See https://github.com/python/mypy/issues/2783#issuecomment-579868936
    @abstractmethod
    def __eq__(self, other: Connection) -> bool:  # type: ignore[override]
        """Returns True if ``self`` is logically equal to ``other``."""

    @abstractmethod
    def __getitem__(self, key: Hashable) -> Any:
        """Supports index accessor notation to retrieve values from the ``attr`` dictionary."""

    @abstractmethod
    def __setitem__(self, key: Hashable, value: Any) -> None:
        """Supports index accessor notation to set values in the ``attr`` dictionary."""

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Connection:
            return abc_utils.check_methods(C, "__eq__", "__getitem__", "__setitem__", "attr",
                "is_loop", "label", "vertex1", "vertex2", "weight")
        return NotImplemented

    @property
    @abstractmethod
    def attr(self) -> Dict[Hashable, Any]:
        """Attribute dictionary to store optional data associated with a connection."""

    @abstractmethod
    def has_attributes_dict(self) -> bool:
        """Returns True if this connection has an instantiated ``attr`` dictionary. Since the
        ``attr`` dictionary is only created as needed, this method can be used to save memory.
        Calling an ``attr`` accessor (such as the ``attr`` property), results in automatic
        dictionary instantiation."""

    @abstractmethod
    def is_loop(self) -> bool:
        """Returns True if the connection links a vertex to itself."""

    @property
    @abstractmethod
    def label(self) -> str:
        """A consistent string representation that should, at a minimum, include the vertex
        endpoints."""

    @property
    @abstractmethod
    def vertex1(self) -> V:
        """The first vertex."""

    @property
    @abstractmethod
    def vertex2(self) -> V:
        """The second vertex."""

    @property
    @abstractmethod
    def weight(self) -> float:
        """The connection weight."""


class MultiConnection(ABC, Generic[V]):
    """A connection that may include multiple parallel connections between two vertices in a
    graph."""

    __slots__ = ()

    # mypy: See https://github.com/python/mypy/issues/2783#issuecomment-579868936
    @abstractmethod
    def __eq__(self, other: MultiConnection) -> bool:  # type: ignore[override]
        """Returns True if ``self`` is logically equal to ``other``."""

    @classmethod
    def __subclasshook__(cls, C):
        if cls is MultiConnection:
            return abc_utils.check_methods(C, "__eq__", "add_connection", "connections",
                "get_connection", "is_loop", "label", "multiplicity", "vertex1", "vertex2",
                "weight")
        return NotImplemented

    @abstractmethod
    def add_connection(
        self, weight: float = DEFAULT_WEIGHT, key: ConnectionKey = DEFAULT_CONNECTION_KEY, **attr
    ) -> Connection[V]:
        """Adds a new connection to this multiconnection.

        Args:
            weight: Optional; The connection weight. Defaults to ``DEFAULT_WEIGHT`` (1.0).
            key: Optional; The key to associate with the new connection that distinguishes it from
                other parallel connections in the multiedge.
            **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

        Returns:
            Connection[V]: The newly added connection.
        """

    @abstractmethod
    def connections(self) -> Iterator[Connection[V]]:
        """An iterator over the connections in the multiconnection."""

    @abstractmethod
    def connection_items(self) -> Iterator[Tuple[ConnectionKey, Connection[V]]]:
        """An iterator over the connection keys and their associated connections in the
        multiconnection."""

    @abstractmethod
    def get_connection(self, key: ConnectionKey) -> Connection[V]:
        """Retrieves a connection by its key."""

    @abstractmethod
    def is_loop(self) -> bool:
        """Returns True if the connection links a vertex to itself."""

    @property
    @abstractmethod
    def label(self) -> str:
        """A consistent string representation that should, at a minimum, include the vertex
        endpoints."""

    @property
    @abstractmethod
    def multiplicity(self) -> int:
        """The multiplicity is the number of connections within the multiedge."""

    @property
    @abstractmethod
    def vertex1(self) -> V:
        """The first vertex."""

    @property
    @abstractmethod
    def vertex2(self) -> V:
        """The second vertex."""

    @property
    def weight(self) -> float:
        """The weight of the multiconnection, including parallel connections."""


class _EdgeConnectionData:
    """Unique data associated with a connection in a multiedge, such as a weight and custom
    attributes.

    Args:
        weight: Optional; Edge weight. Defaults to ``DEFAULT_WEIGHT`` (1.0).
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.
    """

    __slots__ = ("_attr", "weight")

    def __init__(self, weight: float = DEFAULT_WEIGHT, **attr):
        self._attr: Optional[dict] = None  # Initialized lazily using property getter.
        for k, v in attr.items():
            self.attr[k] = v

        self.weight = weight

    @property
    def attr(self) -> dict:
        """Attribute dictionary to store optional data associated with a multiedge connection."""
        if not self._attr:
            self._attr = dict()
        return self._attr

    def has_attributes_dict(self) -> bool:
        """Returns True if this connection has an instantiated ``attr`` dictionary."""
        return self._attr is not None


class EdgeViewBase(Connection[V], Generic[V]):
    """A dynamic view of an edge connection. Edge views provide an edge-like API for each of the
    parallel connections in a multiedge.

    Args:
        multiconnection: A multiconnection object representing multiple parallel edge connections.
        edge_data: The data associated with the particular edge connection that this edge view
            represents.
    """

    __slots__ = ("edge_data", "multiconnection")

    def __init__(self, multiconnection: MultiConnection[V], edge_data: _EdgeConnectionData):
        self.multiconnection = multiconnection
        self.edge_data = edge_data

    def __eq__(self, other: EdgeViewBase[V]) -> bool:  # type: ignore[override]
        """Returns True if ``self`` is logically equal to ``other``."""
        if isinstance(other, Connection):
            return _are_connections_equal(self, other)
        return NotImplemented  # Delegate equality check to the right-hand side.

    def __getitem__(self, key: Hashable) -> Any:
        """Supports index accessor notation to retrieve values from the ``attr`` dictionary."""
        return self.edge_data.attr[key]

    def __repr__(self) -> str:
        return self.__str__()

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """Supports index accessor notation to set values in the ``attr`` dictionary."""
        self.edge_data.attr[key] = value

    def __str__(self) -> str:
        return f"{self.__class__.__name__}{self.multiconnection.label}"

    @property
    def attr(self) -> Dict[Hashable, Any]:
        """Attribute dictionary to store optional data associated with an edge."""
        return self.edge_data.attr

    def has_attributes_dict(self) -> bool:
        """Returns True if this connection has an instantiated ``attr`` dictionary. Since the
        ``attr`` dictionary is only created as needed, this method can be used to save memory.
        Calling an ``attr`` accessor (such as the ``attr`` property), results in automatic
        dictionary instantiation."""
        return self.edge_data.has_attributes_dict()

    def is_loop(self) -> bool:
        """Returns True if this edge connects a vertex to itself."""
        return self.multiconnection.is_loop()

    @property
    def label(self) -> str:
        """A consistent string representation."""
        return self.multiconnection.label

    @property
    @abstractmethod
    def vertex1(self) -> V:
        """The first vertex. For DiEdge objects, this is a synonym for the ``tail`` property."""

    @property
    @abstractmethod
    def vertex2(self) -> V:
        """The second vertex. For DiEdge objects, this is a synonym for the ``head`` property."""

    @property
    def weight(self) -> float:
        """The edge weight."""
        return self.edge_data.weight


class EdgeView(EdgeViewBase[MultiVertex]):
    """A dynamic view of an undirected edge. Edge views provide an edge-like API for each of the
    parallel edge connections in a multiedge.

    Args:
        multiconnection: A multiconnection object representing multiple parallel edge connections.
        edge_data: The data associated with the particular edge connection that this edge view
            represents.
    """

    __slots__ = ()

    @property
    def vertex1(self) -> MultiVertex:
        """The first vertex."""
        return self.multiconnection.vertex1

    @property
    def vertex2(self) -> MultiVertex:
        """The second vertex."""
        return self.multiconnection.vertex2


class DiEdgeView(EdgeViewBase[MultiDiVertex]):
    """A dynamic view of a directed edge. Edge views provide an edge-like API for each of the
    parallel edge connections in a multiedge.

    Args:
        multiconnection: A multiconnection object representing multiple parallel edge connections.
        edge_data: The data associated with the particular edge connection that this edge view
            represents.
    """

    __slots__ = ()

    @property
    def head(self) -> MultiDiVertex:
        """The head vertex, which is the destination of the directed edge."""
        return self.vertex2

    @property
    def tail(self) -> MultiDiVertex:
        """The tail vertex, which is the origin of the directed edge."""
        return self.vertex1

    @property
    def vertex1(self) -> MultiDiVertex:
        """The first vertex. This is a synonym for the ``tail`` property."""
        return self.multiconnection.vertex1

    @property
    def vertex2(self) -> MultiDiVertex:
        """The second vertex. This is a synonym for the ``head`` property."""
        return self.multiconnection.vertex2


class EdgeBase(Connection[V], Generic[V]):
    """Abstract base class from which all single-connection edge classes inherit.

    Args:
        vertex1: The first vertex. In undirected edges, the order of ``vertex1`` and ``vertex2``
            does not matter. For subclasses implementing directed edges, it is recommended to
            rename these arguments ``tail`` (for ``vertex1``) and ``head`` (for ``vertex2``).
        vertex2: The second vertex.
        weight: Optional; Edge weight. Defaults to ``DEFAULT_WEIGHT`` (1.0).
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.
    """

    __slots__ = ("_attr", "_label", "_parent_graph", "_vertex1", "_vertex2", "_weight")

    def __init__(self, vertex1: V, vertex2: V, weight: float = DEFAULT_WEIGHT, **attr):
        self._vertex1 = vertex1
        self._vertex2 = vertex2

        self._label = create_edge_label(vertex1, vertex2,
            is_directed=vertex1._parent_graph.is_directed())

        self._parent_graph: GraphBase = vertex1._parent_graph
        self._weight = weight

        self._attr: Optional[dict] = None  # Initialized lazily using property getter.
        for k, v in attr.items():
            self.attr[k] = v

    def __eq__(self, other) -> bool:
        if isinstance(other, Connection):
            return _are_connections_equal(self, other)
        return NotImplemented  # Delegate equality check to the right-hand side.

    def __getitem__(self, key: Hashable) -> Any:
        """Supports index accessor notation to retrieve values from the ``attr`` dictionary."""
        return self.attr[key]

    def __hash__(self) -> int:
        """Creates a hash key using the edge label."""
        return hash(self.label)

    def __repr__(self) -> str:
        return self.__str__()

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """Supports index accessor notation to set values in the ``attr`` dictionary."""
        self.attr[key] = value

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are lexicographically sorted.
        """

    @property
    def attr(self) -> Dict[Hashable, Any]:
        """Attribute dictionary to store optional data associated with an edge."""
        if not self._attr:
            self._attr = dict()
        return self._attr

    def contract(self, remove_loops: bool = False) -> None:
        """Contracts this edge by removing it from the graph and merging its two incident
        vertices.

        Edge contraction is written as :math:`G/e`, which should not be confused with set
        difference, written as :math:`B \setminus A = \{ x\in B \mid x \notin A \}`.

        For a formal definition, see Wikipedia, `"Edge contraction"
        <https://en.wikipedia.org/wiki/Edge_contraction>`_ [WEC2020]_.

        For efficiency, only one of the two incident vertices is actually deleted. After the edge
        contraction:

           - Incident edges of ``vertex2`` are modified such that ``vertex2`` is replaced by
             ``vertex1``
           - Incident loops on ``vertex2`` become loops on ``vertex1``
           - ``vertex2`` is deleted from the graph

        When ``vertex2`` is deleted, its incident edges are also deleted.

        In some cases, an incident edge of ``vertex2`` will be modified such that by replacing
        ``vertex2`` with ``vertex1``, there exists an edge in the graph matching the new endpoints.
        In this case, the existing edge is not modified.

        If the graph does not contain an edge matching the new endpoints after replacing ``vertex2``
        with ``vertex1``, then a new edge object is added to the graph.

        Note that if either ``GraphBase._allow_self_loops`` is False or ``remove_loops`` is True,
        self loops will be deleted from the merged vertex (``vertex1``).

        Args:
            remove_loops: If True, loops on the merged vertices will be removed. Defaults to False.

        References:
         .. [WEC2020] Wikipedia contributors. "Edge contraction." Wikipedia, The Free
                      Encyclopedia. Available from: https://en.wikipedia.org/wiki/Edge_contraction.
                      Accessed 19 October 2020.
        """
        _contract_connection(self, remove_loops)

    def has_attributes_dict(self) -> bool:
        """Returns True if this connection has an instantiated ``attr`` dictionary. Since the
        ``attr`` dictionary is only created as needed, this method can be used to save memory.
        Calling an ``attr`` accessor (such as the ``attr`` property), results in automatic
        dictionary instantiation."""
        return self._attr is not None

    def is_loop(self) -> bool:
        """Returns True if this edge connects a vertex to itself."""
        return self._vertex1.label == self._vertex2.label

    @property
    def label(self) -> str:
        """A consistent string representation that includes the vertex endpoints. Directed edges
        have labels with the vertices ordered based on the initialization order. Undirected edges
        have labels with vertices lexicographically sorted to ensures consistency. For example,
        both :math:`(1, 2)` and :math:`(2, 1)` refer to the same undirected edge, but the edge
        label would always be "(1, 2)".
        """
        return self._label

    def remove(self, remove_isolated_vertices: bool = False) -> None:
        """Removes this edge from the graph.

        Args:
            remove_isolated_vertices: If True, then vertices adjacent to ``edge`` that become
                isolated after the edge removal are also removed. Defaults to False.
        """
        self._parent_graph.remove_edge(self, remove_isolated_vertices)

    @property
    def vertex1(self) -> V:
        """The first vertex. For DiEdge objects, this is a synonym for the ``tail`` property."""
        return self._vertex1

    @property
    def vertex2(self) -> V:
        """The second vertex. For DiEdge objects, this is a synonym for the ``head`` property."""
        return self._vertex2

    @property
    def weight(self) -> float:
        """The edge weight."""
        return self._weight


class MultiEdgeBase(MultiConnection[V], Generic[V]):
    """Abstract base class from which all multiedge classes inherit.

    Args:
        vertex1: The first vertex. In undirected multiedges, the order of ``vertex1`` and
            ``vertex2`` does not matter. For subclasses implementing directed edges, it is
            recommended to rename these arguments ``tail`` (for ``vertex1``) and ``head`` (for
            ``vertex2``).
        vertex2: The second vertex.
        weight: Optional; Edge weight. Defaults to ``DEFAULT_WEIGHT`` (1.0).
        key: Optional; The key to use within the multiedge to reference the new edge connection.
            Each of the potentially multiple parallel connections is assigned a key unique to the
            multiedge object. Defaults to ``DEFAULT_CONNECTION_KEY`` (0).
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary of the first
            edge connection in the new multiedge.
    """

    __slots__ = ("_connections", "_label", "_parent_graph", "_vertex1", "_vertex2", "_weight")

    def __init__(
        self, vertex1: V, vertex2: V, weight: float = DEFAULT_WEIGHT,
        key: ConnectionKey = DEFAULT_CONNECTION_KEY, **attr
    ) -> None:
        self._vertex1 = vertex1
        self._vertex2 = vertex2

        edge_connection = _EdgeConnectionData(weight, **attr)
        self._connections: Dict[ConnectionKey, _EdgeConnectionData] = dict()
        self._connections[key] = edge_connection

        self._label = create_edge_label(vertex1, vertex2,
            is_directed=vertex1._parent_graph.is_directed())
        self._parent_graph: GraphBase = vertex1._parent_graph

    def __eq__(self, other) -> bool:
        if isinstance(other, MultiConnection):
            return _are_multiconnections_equal(self, other)
        return NotImplemented  # Delegate equality check to the RHS.

    def __hash__(self) -> int:
        """Creates a hash key using the edge label."""
        return hash(self.label)

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are lexicographically sorted.
        """

    def add_connection(
        self, weight: float = DEFAULT_WEIGHT, key: Optional[ConnectionKey] = None, **attr
    ) -> EdgeViewBase[V]:
        """Adds a new edge connection to this multiedge. If the connection key already exists, then
        the existing connection key data is replaced with ``weight`` and ``**attr``.

        Args:
            weight: Optional; The connection weight. Defaults to ``DEFAULT_WEIGHT`` (1.0).
            key: Optional; The key to associate with the new connection that distinguishes it from
                other parallel connections in the multiedge.
            **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

        Returns:
            EdgeViewBase[V]: The newly added edge connection.
        """
        new_key = key
        if key is not None and key in self._connections:
            edge_data: _EdgeConnectionData = self._connections[key]
            if edge_data.has_attributes_dict():
                edge_data._attr.clear()
            for k, v in attr.items():
                edge_data.attr[k] = v
            edge_data.weight = weight
        else:
            edge_data = _EdgeConnectionData(weight, **attr)
            if key is None:
                new_key = _create_connection_key(self._connections.keys())

        self._connections[new_key] = edge_data
        return EdgeView(self, self._connections[new_key])

    @abstractmethod
    def connections(self) -> Iterator[EdgeViewBase[V]]:
        """An iterator over the edge connections in the multiconnection."""

    @abstractmethod
    def connection_items(self) -> Iterator[Tuple[ConnectionKey, EdgeViewBase[V]]]:
        """An iterator over the connection keys and their associated connections in the
        multiedge.

        Yields:
            Tuple[ConnectionKey, EdgeViewBase[V]]: Yields a tuple containing the connection key
            and the edge view.
        """

    def contract(self, remove_loops: bool = False) -> None:
        """Contracts this multiedge by removing it from the graph and merging its two incident
        vertices.

        Edge contraction is written as :math:`G/e`, which should not be confused with set
        difference, written as :math:`B \setminus A = \{ x\in B \mid x \notin A \}`.

        For a formal definition, see Wikipedia, `"Edge contraction"
        <https://en.wikipedia.org/wiki/Edge_contraction>`_ [WEC2020]_.

        For efficiency, only one of the two incident vertices is actually deleted. After the edge
        contraction:

           - Incident edges of ``vertex2`` are modified such that ``vertex2`` is replaced by
             ``vertex1``
           - Incident loops on ``vertex2`` become loops on ``vertex1``
           - ``vertex2`` is deleted from the graph
           - If loops are not deleted, then :math:`degree(vertex1)` [post-merge]
             :math:`\\Longleftrightarrow degree(vertex1) + degree(vertex2)` [pre-merge]

        When ``vertex2`` is deleted, its incident edges are also deleted.

        In some cases, an incident edge of ``vertex2`` will be modified such that by replacing
        ``vertex2`` with ``vertex1``, there exists an edge in the graph matching the new endpoints.
        In this case, the multiplicity of the existing multiedge is increased accordingly.

        If the graph does not contain an edge matching the new endpoints after replacing ``vertex2``
        with ``vertex1``, then a new multiedge object is added to the graph.

        Note that if either ``GraphBase._allow_self_loops`` is False or ``remove_loops`` is True,
        self loops will be deleted from the merged vertex (``vertex1``).

        Args:
            remove_loops: If True, loops on the merged vertices will be removed. Defaults to False.
        """
        _contract_multiconnection(self, remove_loops)

    def get_connection(self, key: ConnectionKey) -> EdgeViewBase[V]:
        """Supports index accessor notation to retrieve a multiedge connection by its key."""
        if key in self._connections:
            return DiEdgeView(self, self._connections[key])
        raise KeyError(key)

    def is_loop(self) -> bool:
        """Returns True if this edge connects a vertex to itself."""
        return self._vertex1.label == self._vertex2.label

    @property
    def label(self) -> str:
        """A consistent string representation that includes the vertex endpoints. Directed edges
        have labels with the vertices ordered based on the initialization order. Undirected edges
        have labels with vertices lexicographically sorted to ensures consistency. For example,
        both :math:`(1, 2)` and :math:`(2, 1)` refer to the same undirected edge, but the edge
        label would always be "(1, 2)".
        """
        return self._label

    @property
    def multiplicity(self) -> int:
        """The number of connections within the multiedge.

        For multiedges without parallel connections, the multiplicity is 1. Each parallel edge adds
        1 to the multiplicity.
        """
        return len(self._connections)

    def remove(self, remove_isolated_vertices: bool = False) -> None:
        """Removes this edge from the graph.

        Args:
            remove_isolated_vertices: If True, then vertices adjacent to ``edge`` that become
                isolated after the edge removal are also removed. Defaults to False.
        """
        self._parent_graph.remove_edge(self, remove_isolated_vertices)

    def remove_connection(self, key: ConnectionKey) -> None:
        """Removes an edge connection from this multiedge based on its key. If the multiedge only
        has one connection, then removing the connection is the same as calling :meth:`remove`."""
        if key in self._connections:
            if self.multiplicity > 1:
                self._connections.pop(key)
            else:
                self.remove()
        else:
            raise KeyError(key)

    @property
    def vertex1(self) -> V:
        """The first vertex. For DiEdge objects, this is a synonym for the ``tail`` property."""
        return self._vertex1

    @property
    def vertex2(self) -> V:
        """The second vertex. For DiEdge objects, this is a synonym for the ``head`` property."""
        return self._vertex2

    @property
    def weight(self) -> float:
        """The weight of the multiedge, including parallel connections."""
        return sum(connection.weight for connection in self._connections.values())


class Edge(EdgeBase[Vertex]):
    """An undirected edge that does not allow parallel connections between its vertices.

    To help ensure the integrity of graphs, the ``Edge`` class is abstract and cannot be
    instantiated directly. To create edges, use :meth:`Graph.add_edge
    <vertizee.classes.graph.Graph.add_edge>` and :meth:`Graph.add_edges_from
    <vertizee.classes.graph.Graph.add_edges_from>`.

    Note:
        In an undirected graph, edges :math:`(s, t)` and :math:`(t, s)` represent the same edge.
        Therefore, attempting to add :math:`(s, t)` and :math:`(t, s)` would raise an exception,
        since ``Edge`` objects do not support parallel connections. For parallel connection support,
        see :class:`MultiEdge` and :class:`MultiDiEdge`.
    """

    __slots__ = ()

    def __init__(self, vertex1: Vertex, vertex2: Vertex, weight: float = DEFAULT_WEIGHT, **attr):
        super().__init__(vertex1, vertex2, weight=weight, **attr)

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge that shows the vertex labels, and for
        weighted graphs, includes the edge weight. The vertices are lexicographically sorted.

        Example:
            "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]
        """


class DiEdge(EdgeBase[DiVertex]):
    """A directed edge that does not allow parallel connections between its vertices.

    To help ensure the integrity of graphs, the ``DiEdge`` class is abstract and cannot be
    instantiated directly. To create directed edges, use :meth:`DiGraph.add_edge
    <vertizee.classes.graph.DiGraph.add_edge>` and :meth:`DiGraph.add_edges_from
    <vertizee.classes.graph.DiGraph.add_edges_from>`.

    Note:
        In a directed graph, edge :math:`(s, t)` is distinct from edge :math:`(t, s)`. Adding
        these two edges to a directed graph results in separate ``DiEdge`` objects. However, adding
        the edges to an undirected graph (for example, :class:`Graph
        <vertizee.classes.graph.Graph>`), would raise an exception, since parallel edges are not
        allowed.

    Attributes:
        attr: Attribute dictionary to store optional data associated with the edge.
    """

    __slots__ = ()

    def __init__(self, tail: DiVertex, head: DiVertex, weight: float = DEFAULT_WEIGHT, **attr):
        super().__init__(vertex1=tail, vertex2=head, weight=weight, **attr)

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge that shows the vertex labels, and for
        weighted graphs, includes the edge weight. The vertex order matches the initialization
        order.

        Example:
            "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]
        """

    @property
    def head(self) -> DiVertex:
        """The head vertex, which is the destination of the directed edge."""
        return self._vertex2

    @property
    def tail(self) -> DiVertex:
        """The tail vertex, which is the origin of the directed edge."""
        return self._vertex1


class MultiEdge(MultiEdgeBase[Vertex]):
    """Undirected multiedge that allows multiple parallel connections between its two vertices.

    To help ensure the integrity of graphs, the ``MultiEdge`` class is abstract and cannot be
    instantiated directly. To create ``MultiEdge`` objects, use :meth:`MultiGraph.add_edge
    <vertizee.classes.graph.MultiGraph.add_edge>` and :meth:`MultiGraph.add_edges_from
    <vertizee.classes.graph.MultiGraph.add_edges_from>`.

    Note:
        In an undirected multigraph, edges :math:`(s, t)` and :math:`(t, s)` represent the same
        multiedge. If multiple edges are added with the same vertices, then a single ``MultiEdge``
        instance is used to store the parallel connections. When working with multiedges, use the
        ``multiplicity`` property to determine if the edge represents more than one edge connection.
    """

    __slots__ = ()

    def __init__(
        self, vertex1: Vertex, vertex2: Vertex, weight: float = DEFAULT_WEIGHT,
        key: ConnectionKey = DEFAULT_CONNECTION_KEY, **attr
    ) -> None:
        super().__init__(vertex1, vertex2, weight=weight, key=key, **attr)

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge that shows the vertex labels, and for
        weighted graphs, includes the edge weight. The vertices are lexicographically sorted. The
        string will show separate vertex tuples for each parallel connection, such as
        "(a, b), (a, b), (a, b)".

        Example:
            "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]
        """

    @abstractmethod
    def connections(self) -> Iterator[EdgeView]:
        """An iterator over the edge connections in the multiedge."""

    @abstractmethod
    def connection_items(self) -> Iterator[Tuple[ConnectionKey, EdgeView]]:
        """An iterator over the connection keys and their associated edge connections in the
        multiedge.

        Yields:
            Tuple[ConnectionKey, EdgeView]: Yields a tuple containing the connection key
            and the edge view.
        """


class MultiDiEdge(MultiEdgeBase[DiVertex]):
    """Directed multiedge that allows multiple directed connections between its two vertices.

    To help ensure the integrity of graphs, ``MultiDiEdge`` is abstract and cannot be instantiated
    directly. To create edges, use :meth:`MultiDiGraph.add_edge
    <vertizee.classes.graph.MultiDiGraph.add_edge>` and :meth:`MultiDiGraph.add_edges_from
    <vertizee.classes.graph.MultiDiGraph.add_edges_from>`.

    Note:
        In a directed graph, edge :math:`(s, t)` is distinct from edge :math:`(t, s)`. Adding
        these two edges to a directed multigraph results in separate ``MultiDiEdge`` objects.
        However, adding the edges to an undirected multigraph (for example, :class:`MultiGraph
        <vertizee.classes.graph.MultiGraph>`), results in one :class:`MultiEdge` object with a
        parallel connection.
    """

    __slots__ = ("_connections",)

    def __init__(
        self, tail: DiVertex, head: DiVertex, weight: float = DEFAULT_WEIGHT,
        key: ConnectionKey = DEFAULT_CONNECTION_KEY, **attr
    ):
        super().__init__(vertex1=tail, vertex2=head, weight=weight, key=key, **attr)

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge that shows the vertex labels, and for
        weighted graphs, includes the edge weight. The vertex order matches the initialization
        order. The string will show separate vertex tuples for each parallel connection, such as
        "(a, b), (a, b), (a, b)".

        Example:
            "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]
        """

    @abstractmethod
    def connections(self) -> Iterator[DiEdgeView]:
        """An iterator over the connections in the directed multiedge."""

    @abstractmethod
    def connection_items(self) -> Iterator[Tuple[ConnectionKey, DiEdgeView]]:
        """An iterator over the connection keys and their associated connections in the
        directed multiedge.

        Yields:
            Tuple[ConnectionKey, DiEdgeView]: Yields a tuple containing the connection key
            and the directed edge view.
        """

    @property
    def head(self) -> DiVertex:
        """The head vertex, which is the destination of the directed edge."""
        return self._vertex2

    @property
    def tail(self) -> DiVertex:
        """The tail vertex, which is the origin of the directed edge."""
        return self._vertex1


class _Edge(Edge):
    """Concrete implementation of the abstract :class:`Edge` class."""

    __slots__ = ()

    def __str__(self) -> str:
        return _str_for_connection(self._vertex1, self._vertex2, self._weight)


class _DiEdge(DiEdge):
    """Concrete implementation of the abstract :class:`DiEdge` class."""

    __slots__ = ()

    def __str__(self) -> str:
        return _str_for_connection(self._vertex1, self._vertex2, self._weight)


class _MultiEdge(MultiEdge):
    """Concrete implementation of the abstract :class:`MultiEdge` class."""

    __slots__ = ()

    def __str__(self) -> str:
        return _str_for_multiconnection(self._vertex1, self._vertex2, self._connections)

    def connections(self) -> Iterator[EdgeView]:
        """An iterator over the connections in the multiedge."""
        for connection in self._connections.values():
            yield EdgeView(self, connection)

    def connection_items(self) -> Iterator[Tuple[ConnectionKey, EdgeView]]:
        """An iterator over the connection keys and their associated connections in the
        multiedge."""
        for key, connection in self._connections.items():
            view = EdgeView(self, connection)
            yield key, view


class _MultiDiEdge(MultiDiEdge):
    """Concrete implementation of the abstract :class:`MultiDiEdge` class."""

    __slots__ = ()

    def __str__(self) -> str:
        return _str_for_multiconnection(self._vertex1, self._vertex2, self._connections)

    def connections(self) -> Iterator[DiEdgeView]:
        """An iterator over the connections in the directed multiedge."""
        for connection in self._connections.values():
            yield DiEdgeView(self, connection)

    def connection_items(self) -> Iterator[Tuple[ConnectionKey, DiEdgeView]]:
        """An iterator over the connection keys and their associated connections in the
        directed multiedge."""
        for key, connection in self._connections.items():
            view = DiEdgeView(self, connection)
            yield key, view
