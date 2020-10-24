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

* :class:`DiEdge` - A directed connection between two vertices that
  defines the ``tail`` as the starting vertex and the ``head`` as the destination vertex. Parallel
  connections are not allowed.
* :class:`Edge` - An undirected connection between two vertices. The order of the vertices does not
  matter. However, the string representation will always show vertices sorted in lexicographic
  order. Parallel connections are not allowed.
* :class:`EdgeType` - A type alias defined as
  Union[DiEdge, Edge, EdgeLiteral, MultiDiEdge, MultiEdge] and where EdgeLiteral is an alias for
  various edge-tuple formats.
* :class:`MultiDiEdge` - A directed connection between two vertices that defines the ``tail`` as
  the starting vertex and the ``head`` as the destination vertex. Multi-edges support parallel
  connections.
* :class:`MultiEdge` - An undirected connection between two vertices. The order of the vertices
  does not matter. However, the string representation will always show vertices sorted in
  lexicographic order. Multi-edges support parallel connections.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import collections
from typing import (
    Any, Dict, Final, Generic, Hashable, Iterator, List, Mapping, Optional, Tuple, TYPE_CHECKING,
    TypeVar, Union
)

from vertizee.classes import vertex as vertex_module
from vertizee.utils import abc_utils

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.graph_base import GraphBase
    from vertizee.classes.vertex import DiVertex, Vertex, VertexClass, VertexType

# Type aliases
AttributesDict = dict
Weight = float
EdgeClass = ["DiEdge", "Edge", "MultiDiEdge", "MultiEdge"]
EdgeTuple = Tuple["VertexType", "VertexType"]
EdgeTupleWeighted = Tuple["VertexType", "VertexType", Weight]
EdgeTupleAttr = Tuple["VertexType", "VertexType", AttributesDict]
EdgeTupleWeightedAttr = Tuple["VertexType", "VertexType", Weight, AttributesDict]
EdgeLiteral = Union[EdgeTuple, EdgeTupleWeighted, EdgeTupleAttr, EdgeTupleWeightedAttr]

#: EdgeType: A type alias defined as Union[EdgeClass, EdgeLiteral] and
# where EdgeLiteral is an alias for various edge-tuple formats.
EdgeType = Union[EdgeClass, EdgeLiteral]

# Generic type parameters
V = TypeVar("V", "DiVertex", "Vertex")

DEFAULT_WEIGHT: Final = 1.0
DEFAULT_ATTR_KEY: Final = 0


def create_edge_label(vertex1: "VertexType", vertex2: "VertexType", is_directed: bool) -> str:
    """Creates a string representation of the edge connecting ``vertex1`` and ``vertex2``, for
    example "(1, 2)".

    Directed edges have labels with the vertices ordered based on the instantiation order.
    Undirected edges have labels with vertices lexicographically sorted, which provides a consistent
    representation; for example, both :math:`(1, 2)` and :math:`(2, 1)` refer to the same undirected
    edge, but the edge label would always be "(1, 2)".

    Args:
        vertex1: The first vertex of the edge.
        vertex2: The second vertex of the edge.
        is_directed (bool): True indicates a directed edge, False an undirected edge.

    Returns:
        str: The edge key.
    """
    v1_label = vertex_module.get_vertex_label(vertex1)
    v2_label = vertex_module.get_vertex_label(vertex2)

    if not is_directed and v1_label > v2_label:
        return f"({v2_label}, {v1_label})"
    return f"({v1_label}, {v2_label})"


class _EdgeBase(ABC, Generic[V]):
    """Abstract base class from which all edge classes inherit.

    Args:
        vertex1: The first vertex. In undirected edges, such as :class:`Edge` and,
            :class:`MultiEdge`, the order of ``vertex1`` and ``vertex2`` does not matter. For
            classes implementing directed edges, it is recommended to rename these arguments
            ``tail`` (for ``vertex1``) and ``head`` (for ``vertex2``).
        vertex2: The second vertex.
        weight: Optional; Edge weight. Defaults to 1.0.
    """

    __slots__ = ("_label", "_parent_graph", "_vertex1", "_vertex2", "_weight")

    def __init__(self, vertex1: V, vertex2: V, weight: float = DEFAULT_WEIGHT):
        # IMPORTANT: _vertex1 and _vertex2 are used by __hash__(), and must therefore be
        # treated as immutable (read-only).
        self._vertex1 = vertex1
        self._vertex2 = vertex2

        self._label = f"({vertex1.label}, {vertex2.label})"
        if not vertex1._parent_graph.is_directed_graph():
            if vertex1.label > vertex2.label:
                self._label = f"({vertex2.label}, {vertex1.label})"

        self._parent_graph: GraphBase = vertex1._parent_graph
        self._weight = float(weight)

    @abstractmethod
    def __eq__(self, other) -> bool:
        """Returns True if ``other`` equals ``self``."""

    def __hash__(self) -> int:
        """Creates a hash key using the edge's vertices.

        Note that ``__eq__`` is defined to take ``_weight`` and ``_parallel_edge_count`` into
        consideration, whereas ``__hash__`` does not. This is because ``_weight`` and
        ``_parallel_edge_count``are not intended to be immutable throughout the lifetime of the
        object.

        From: "Python Hashes and Equality" <https://hynek.me/articles/hashes-and-equality/>:

        "Hashes can be less picky than equality checks. Since key lookups are always followed by
        an equality check, your hashes don’t have to be unique. That means that you can compute
        your hash over an immutable subset of attributes that may or may not be a unique
        "primary key" for the instance."
        """
        if (isinstance(self._vertex1, vertex_module.Vertex) and
            self.vertex1.label > self.vertex2.label):
            return hash((self.vertex2, self.vertex1))

        return hash((self.vertex1, self.vertex2))

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are lexicographically sorted.
        """

    @classmethod
    def __subclasshook__(cls, C):
        if cls is _EdgeBase:
            return abc_utils.check_methods(C, "__eq__", "__hash__", "__repr__", "__str__", "label",
                "is_loop", "vertex1", "vertex2", "weight")
        return NotImplemented

    @property
    def label(self) -> str:
        """A string representation of the edge that includes the vertex endpoints, for example
        "(1, 2)". Directed edges have labels with the vertices ordered based on the instantiation
        order. Undirected edges have labels with vertices lexicographically sorted, which provides a
        consistent representation; for example, both :math:`(1, 2)` and :math:`(2, 1)` refer to the
        same undirected edge, but the edge label would always be "(1, 2)".
        """
        return self._label

    def is_loop(self) -> bool:
        """Returns True if this edge connects a vertex to itself."""
        return self._vertex1.label == self._vertex2.label

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


class Edge(_EdgeBase["Vertex"]):
    """An undirected edge that does not allow parallel connections between its vertices.

    To help ensure the integrity of graphs, the ``Edge`` class is abstract and cannot be
    instantiated directly. To create edges, use :meth:`Graph.add_edge
    <vertizee.classes.graph.Graph.add_edge>` and :meth:`Graph.add_edges_from
    <vertizee.classes.graph.Graph.add_edges_from>`.

    Note:
        In an undirected graph, edges :math:`(s, t)` and :math:`(t, s)` represent the same edge.
        Therefore, attempting to add :math:`(s, t)` and :math:`(t, s)` would raise an exception,
        since ``Edge`` objects do not support parallel connections. For parallel edge support,
        see :class:`MultiEdge` and :class:`MultiDiEdge`.
    """

    __slots__ = ("_attr",)

    def __init__(self, vertex1: Vertex, vertex2: Vertex, weight: float = DEFAULT_WEIGHT, **attr):
        super().__init__(vertex1, vertex2, weight)

        self._attr: Optional[dict] = None  # Initialized lazily using property getter.
        for k, v in attr.items():
            self.attr[k] = v

    def __eq__(self, other) -> bool:
        if isinstance(other, Edge):
            v1 = self._vertex1
            v2 = self._vertex2
            o_v1 = other._vertex1
            o_v2 = other._vertex2
            if v1.label > v2.label:
                v1, v2 = v2, v1
            if o_v1.label > o_v2.label:
                o_v1, o_v2 = o_v2, o_v1

            if v1 != o_v1 or v2 != o_v2 or self._weight != other._weight:
                return False
            return True
        return NotImplemented  # Delegate equality check to the RHS.

    def __getitem__(self, key: Hashable) -> Any:
        """Supports index accessor notation to retrieve values from the ``attr`` dictionary."""
        return self.attr[key]

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

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Edge:
            return abc_utils.check_methods(C, "__eq__", "__getitem__", "__hash__", "__setitem__",
                "__str__", "attr", "label", "is_loop", "vertex1", "vertex2", "weight")
        return NotImplemented

    @property
    def attr(self) -> dict:
        """Attribute dictionary to store ad hoc data associated with an edge."""
        if not self._attr:
            self._attr = dict()
        return self._attr


class DiEdge(_EdgeBase["DiVertex"]):
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
        attr: Attribute dictionary to store ad hoc data associated with the edge.
    """

    __slots__ = ("_attr",)

    def __init__(self, tail: DiVertex, head: DiVertex, weight: float = DEFAULT_WEIGHT, **attr):
        super().__init__(vertex1=tail, vertex2=head, weight=weight)

        self._attr: Optional[dict] = None  # Initialized lazily using property getter.
        for k, v in attr.items():
            self.attr[k] = v

    def __eq__(self, other) -> bool:
        if isinstance(other, DiEdge):
            v1 = self._vertex1
            v2 = self._vertex2
            o_v1 = other._vertex1
            o_v2 = other._vertex2

            if v1 != o_v1 or v2 != o_v2 or self._weight != other._weight:
                return False
            return True
        return NotImplemented  # Delegate equality check to the RHS.

    def __getitem__(self, key: Hashable) -> Any:
        """Supports index accessor notation to retrieve values from the ``attr`` dictionary."""
        return self.attr[key]

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

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Edge:
            return abc_utils.check_methods(C, "__eq__", "__getitem__", "__hash__", "__setitem__",
                "__str__", "attr", "head", "label", "is_loop", "tail", "vertex1", "vertex2",
                "weight")
        return NotImplemented

    @property
    def attr(self) -> dict:
        """Attribute dictionary to store ad hoc data associated with an edge."""
        if not self._attr:
            self._attr = dict()
        return self._attr

    @property
    def head(self) -> DiVertex:
        """The head vertex, which is the destination of the directed edge."""
        return self._vertex2

    @property
    def tail(self) -> DiVertex:
        """The tail vertex, which is the origin of the directed edge."""
        return self._vertex1


class MultiEdge(_EdgeBase["Vertex"]):
    """An undirected edge that allows multiple parallel connections between its vertices.

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

    __slots__ = ("_connections",)

    def __init__(
        self, vertex1: Vertex, vertex2: Vertex, weight: float = DEFAULT_WEIGHT,
        key: Hashable = DEFAULT_ATTR_KEY, **attr
    ):
        super().__init__(vertex1, vertex2)

        connection = _MultiEdgeConnection(weight, **attr)
        self._connections: Dict[str, _MultiEdgeConnection] = dict()
        self._connections[key] = connection

    def __eq__(self, other) -> bool:
        if isinstance(other, MultiEdge):
            v1 = self._vertex1
            v2 = self._vertex2
            o_v1 = other._vertex1
            o_v2 = other._vertex2
            if v1.label > v2.label:
                v1, v2 = v2, v1
            if o_v1.label > o_v2.label:
                o_v1, o_v2 = o_v2, o_v1

            if v1 != o_v1 or v2 != o_v2:
                return False
            if self._connections != other._connections:
                return False
            return True
        return NotImplemented  # Delegate equality check to the RHS.

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are lexicographically sorted. In multigraphs, the string will show separate vertex
        tuples for each parallel connection, such as "(a, b), (a, b), (a, b)" for a multiedge with
        multiplicity 3.
        """

    @classmethod
    def __subclasshook__(cls, C):
        if cls is MultiEdge:
            return abc_utils.check_methods(C, "__eq__", "__getitem__", "__hash__", "__setitem__",
                "__str__", "attr", "is_loop", "label", "multiplicity", "vertex1", "vertex2",
                "weight")
        return NotImplemented

    def connections(self) -> Iterator[Tuple[Hashable, MultiEdgeConnectionView]]:
        """An iterator providing dynamic views of the connections in the multiedge.

        Yields:
            Tuple[Hashable, MultiEdgeConnectionView]: Yields a tuple containing the connection key
            and the multiedge connection view.
        """
        for key, connection in self._connections:
            # Do not access connection.attr unless needed to avoid lazy initialization.
            if connection.has_attributes():
                view = MultiEdgeConnectionView(
                    self._vertex1, self._vertex2, connection.weight, **connection.attr)
            else:
                view = MultiEdgeConnectionView(self._vertex1, self._vertex2, connection.weight)
            yield key, view

    @property
    def multiplicity(self) -> int:
        """The multiplicity is the number of connections within the multiedge.

        For edges without parallel connections, the multiplicity is 1. Each parallel edge adds 1 to
        the multiplicity.
        """
        return len(self._connections)

    @property
    def weight(self) -> float:
        """The weight of the multiedge, including parallel connections.

        Returns:
            float: The total multiedge weight, including parallel edges.
        """
        return sum(connection.weight for connection in self._connections.values())


class MultiDiEdge(_EdgeBase["DiVertex"]):
    """Edge that supports multiple directed connections between two vertices.

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
        key: Hashable = DEFAULT_ATTR_KEY, **attr
    ):
        super().__init__(vertex1=tail, vertex2=head)

        connection = _MultiEdgeConnection(weight, **attr)
        self._connections: Dict[str, _MultiEdgeConnection] = dict()
        self._connections[key] = connection

    def __eq__(self, other) -> bool:
        if isinstance(other, MultiEdge):
            v1 = self._vertex1
            v2 = self._vertex2
            o_v1 = other._vertex1
            o_v2 = other._vertex2

            if v1 != o_v1 or v2 != o_v2:
                return False
            if self._connections != other._connections:
                return False
            return True
        return NotImplemented  # Delegate equality check to the RHS.

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are lexicographically sorted. In multigraphs, the string will show separate vertex
        tuples for each parallel connection, such as "(a, b), (a, b), (a, b)" for a multiedge with
        multiplicity 3.
        """

    @classmethod
    def __subclasshook__(cls, C):
        if cls is MultiEdge:
            return abc_utils.check_methods(C, "__eq__", "__getitem__", "__hash__", "__setitem__",
                "__str__", "attr", "head", "is_loop", "label", "multiplicity", "tail", "vertex1",
                "vertex2", "weight")
        return NotImplemented

    def connections(self) -> Iterator[Tuple[Hashable, MultiEdgeConnectionView]]:
        """An iterator providing dynamic views of the connections in the multiedge.

        Yields:
            Tuple[Hashable, MultiEdgeConnectionView]: Yields a tuple containing the connection key
            and the multiedge connection view.
        """
        for key, connection in self._connections:
            # Do not access connection.attr unless needed to avoid lazy initialization.
            if connection.has_attributes():
                view = MultiEdgeConnectionView(
                    self._vertex1, self._vertex2, connection.weight, **connection.attr)
            else:
                view = MultiEdgeConnectionView(self._vertex1, self._vertex2, connection.weight)
            yield key, view

    @property
    def head(self) -> DiVertex:
        """The head vertex, which is the destination of the directed edge."""
        return self._vertex2

    @property
    def multiplicity(self) -> int:
        """The multiplicity is the number of connections within the multiedge.

        For edges without parallel connections, the multiplicity is 1. Each parallel edge adds 1 to
        the multiplicity.
        """
        return len(self._connections)

    @property
    def tail(self) -> DiVertex:
        """The tail vertex, which is the origin of the directed edge."""
        return self._vertex1

    @property
    def weight(self) -> float:
        """The weight of the multiedge, including parallel connections.

        Returns:
            float: The total multiedge weight, including parallel edges.
        """
        return sum(connection.weight for connection in self._connections.values())


#
# Concrete implementations.
#

class _MultiEdgeConnection:
    """Class to store unique data for each connection in a multiedge. A multiedge may have multiple
    parallel connections.

    Args:
        weight: Optional; Edge weight. Defaults to 1.0.
        **attr: Optional; Keyword arguments to be added to the ``attr`` dictionary.
    """

    __slots__ = ("_attr", "weight")

    def __init__(self, weight: float = DEFAULT_WEIGHT, **attr):
        self._attr: Optional[dict] = None  # Initialized lazily using property getter.
        for k, v in attr.items():
            self.attr[k] = v

        self.weight: float = float(weight)

    @property
    def attr(self) -> dict:
        """Attribute dictionary to store ad hoc data associated with a multiedge connection."""
        if not self._attr:
            self._attr = dict()
        return self._attr

    def has_attributes(self) -> bool:
        """Returns True if this connection has custom attributes stored in its ``attr``
        dictionary."""
        return self._attr is not None


class MultiEdgeConnectionView(Generic[V]):
    """Lightweight container for viewing each connection in a multiedge as if it were a standalone
    edge."""

    __slots__ = ("_attr", "label", "vertex1", "vertex2", "weight")

    def __init__(self, vertex1: V, vertex2: V, weight: float, **attr):
        self.vertex1 = vertex1
        self.vertex2 = vertex2

        self.label = f"({vertex1.label}, {vertex2.label})"
        if not vertex1._parent_graph.is_directed_graph():
            if vertex1.label > vertex2.label:
                self.label = f"({vertex2.label}, {vertex1.label})"

        self._attr: Optional[dict] = None  # Initialized lazily using property getter.
        for k, v in attr.items():
            self.attr[k] = v

        self.weight = float(weight)

    def __getitem__(self, key: Hashable) -> Any:
        """Supports index accessor notation to retrieve values from the ``attr`` dictionary."""
        return self.attr[key]

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """Supports index accessor notation to set values in the ``attr`` dictionary."""
        self.attr[key] = value

    def __str__(self) -> str:
        if self.vertex1.label > self.vertex2.label:
            edge_str = f"({self.vertex2.label}, {self.vertex1.label}"
        else:
            edge_str = f"({self.vertex1.label}, {self.vertex2.label}"

        if self.vertex1._parent_graph.is_weighted():
            edge_str = f"{edge_str}, {self.weight})"
        else:
            edge_str = f"{edge_str})"
        return edge_str

    @property
    def attr(self) -> dict:
        """Attribute dictionary to store ad hoc data associated with an edge."""
        if not self._attr:
            self._attr = dict()
        return self._attr


class MultiDiEdgeConnectionView(Generic[V]):
    """Lightweight container for viewing each connection in a directed multiedge as if it were a
    standalone edge."""

    __slots__ = ("_attr", "head", "label", "tail", "vertex1", "vertex2", "weight")

    def __init__(self, tail: V, head: V, weight: float, **attr):
        self.vertex1 = tail
        self.vertex2 = head
        self.tail = tail
        self.head = head

        self.label = f"({tail.label}, {head.label})"

        self._attr: Optional[dict] = None  # Initialized lazily using property getter.
        for k, v in attr.items():
            self.attr[k] = v

        self.weight = float(weight)

    def __getitem__(self, key: Hashable) -> Any:
        """Supports index accessor notation to retrieve values from the ``attr`` dictionary."""
        return self.attr[key]

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """Supports index accessor notation to set values in the ``attr`` dictionary."""
        self.attr[key] = value

    def __str__(self) -> str:
        edge_str = f"({self.tail.label}, {self.head.label}"

        if self.vertex1._parent_graph.is_weighted():
            edge_str = f"{edge_str}, {self.weight})"
        else:
            edge_str = f"{edge_str})"
        return edge_str

    @property
    def attr(self) -> dict:
        """Attribute dictionary to store ad hoc data associated with an edge."""
        if not self._attr:
            self._attr = dict()
        return self._attr


class _Edge(Edge):
    """Concrete implementation of the abstract :class:`Edge` class."""

    __slots__ = ()

    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are lexicographically sorted.
        """
        if self.vertex1.label > self.vertex2.label:
            edge_str = f"({self.vertex2.label}, {self.vertex1.label}"
        else:
            edge_str = f"({self.vertex1.label}, {self.vertex2.label}"

        if self._parent_graph.is_weighted():
            edge_str = f"{edge_str}, {self._weight})"
        else:
            edge_str = f"{edge_str})"

        return edge_str


class _DiEdge(DiEdge):
    """Concrete implementation of the abstract :class:`DiEdge` class."""

    __slots__ = ()

    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted].
        """
        edge_str = f"({self.vertex1.label}, {self.vertex2.label}"

        if self._parent_graph.is_weighted():
            edge_str = f"{edge_str}, {self._weight})"
        else:
            edge_str = f"{edge_str})"

        return edge_str


class _MultiEdge(MultiEdge):
    """Concrete implementation of the abstract :class:`MultiEdge` class."""

    __slots__ = ()

    def __eq__(self, other) -> bool:
        if not isinstance(other, MultiEdge):
            return False

        v1 = self.vertex1
        v2 = self.vertex2
        o_v1 = other.vertex1
        o_v2 = other.vertex2
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

    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are lexicographically sorted. In multigraphs, the string will show separate vertex
        tuples for each parallel edge, such as "(a, b), (a, b), (a, b)" for a multiedge with
        multiplicity 3.
        """
        if self.vertex1.label > self.vertex2.label:
            edge_str = f"({self.vertex2.label}, {self.vertex1.label}"
        else:
            edge_str = f"({self.vertex1.label}, {self.vertex2.label}"

        if self.vertex1._parent_graph.is_weighted():
            edge_str = f"{edge_str}, {self._weight})"
        else:
            edge_str = f"{edge_str})"

        edges = [edge_str for _ in range(self.multiplicity)]
        return ", ".join(edges)


class _MultiDiEdge(MultiDiEdge):
    """Concrete implementation of the abstract :class:`MultiDiEdge` class."""

    __slots__ = ()

    def __eq__(self, other) -> bool:
        if not isinstance(other, MultiDiEdge):
            return False

        v1 = self.vertex1
        v2 = self.vertex2
        o_v1 = other.vertex1
        o_v2 = other.vertex2
        if (
            v1 != o_v1
            or v2 != o_v2
            or self._parallel_edge_count != other._parallel_edge_count
            or self._weight != other._weight
        ):
            return False
        return True

    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are lexicographically sorted. In multigraphs, the string will show separate vertex
        tuples for each parallel edge, such as "(a, b), (a, b), (a, b)" for a multiedge with
        multiplicity 3.
        """
        edge_str = f"({self.vertex1.label}, {self.vertex2.label}"

        if self.vertex1._parent_graph.is_weighted():
            edge_str = f"{edge_str}, {self._weight})"
        else:
            edge_str = f"{edge_str})"

        edges = [edge_str for _ in range(self.multiplicity)]
        return ", ".join(edges)
